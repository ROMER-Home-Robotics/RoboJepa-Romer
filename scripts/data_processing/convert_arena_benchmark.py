#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert Arena sim benchmark rollouts (policy episodes) into the episode format the robo_jepa dataloader reads.

The benchmark writes ``<results_dir>/<benchmark>/<policy>/<NNN_layout_seed>/`` per episode, with ``episode.json``
(task, phase, outcome) and the policy client's recording under ``policy/**/``::

    raw_side_by_side.mp4    # [front | wrist], 2 x 1280x720, one frame per control step (5 Hz)
    steps.jsonl             # per control step: ros_time, eef_pos, eef_quat (x, y, z, w; world frame), gripper_width

Frame ``i`` of ``raw_side_by_side.mp4`` is the observation of step ``i``. Unlike the collection data
(``convert_arena.py``) there is no hand-camera stream of its own, and ``front.mp4`` carries a large text banner, so both
views come from ``raw_side_by_side.mp4``. Its front half has a small debug-text box in the top-left corner (over the
static background); it is inpainted (OpenCV Telea) so the model does not see text the training videos never had.

Output, under ``--dst`` (same layout and ``data.h5`` conventions as ``convert_arena.py``: ``franka`` embodiment, EE
pose in the robot base frame, gripper 0 = open / 1 = closed)::

    dataset_index.csv                       # column "path"
    episodes.csv                            # path, benchmark, policy, task, phase, outcome
    dropped_episodes.tsv
    <benchmark>/<policy>/<episode>/{data.h5, front.mp4, hand.mp4}     # --fps (default 5) uniform timeline

The gripper closedness is ``clip(1 - width / --gripper_open, 0, 1)`` (the sim hand opens to 0.04 per finger).

    python scripts/data_processing/convert_arena_benchmark.py --src ~/arena_sim_data/benchmark/rs1030_* \\
        --dst /tmp/rs1030_benchmark --workers 12

Configs / evals read it with ``raw_fps`` equal to ``--fps``.
"""

import argparse
import csv
import glob
import json
import os
from fractions import Fraction
from multiprocessing import Pool

import av
import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Debug-text box of the front half (pixels of the 1280x720 view): two lines of text in black boxes.
OVERLAY_BOX = (0, 6, 410, 80)  # x0, y0, x1, y1


def find_episodes(root):
    """Episode dirs under a benchmark root: <policy>/<NNN_...>/ with an episode.json and a policy recording."""
    out = []
    for ep in sorted(glob.glob(os.path.join(root, "*", "[0-9][0-9][0-9]_*"))):
        if os.path.isfile(os.path.join(ep, "episode.json")):
            out.append(ep)
    return out


def recording(ep):
    """(raw_side_by_side.mp4, steps.jsonl) of the episode's policy recording, or None."""
    for d in sorted(glob.glob(os.path.join(ep, "policy", "**", "steps.jsonl"), recursive=True)):
        d = os.path.dirname(d)
        mp4 = os.path.join(d, "raw_side_by_side.mp4")
        if os.path.exists(mp4):
            return mp4, os.path.join(d, "steps.jsonl")
    return None


def write_video(frames, dst, fps, crf):
    h, w = frames[0].shape[:2]
    rate = Fraction(fps).limit_denominator(1001)
    with av.open(dst, "w") as out:
        enc = out.add_stream("libx264", rate=rate)
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.options = {"crf": str(crf), "preset": "veryfast", "g": "30"}
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = i, 1 / rate
            for packet in enc.encode(frame):
                out.mux(packet)
        for packet in enc.encode():
            out.mux(packet)


def convert_episode(args):
    ep, out, opts = args
    if not opts["overwrite"] and os.path.exists(os.path.join(out, "data.h5")):
        return None
    try:
        rec = recording(ep)
        if rec is None:
            return "no_recording"
        mp4, steps_path = rec
        rows = [json.loads(line) for line in open(steps_path) if line.strip()]
        rows = [r for r in rows if r.get("eef_pos") is not None and r.get("eef_quat") is not None]
        with av.open(mp4) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        n = min(len(rows), len(frames))
        if abs(len(rows) - len(frames)) > 1:
            return f"frame_step_mismatch:{len(frames)}_frames_{len(rows)}_steps"
        rows, frames = rows[:n], frames[:n]
        t = np.array([r["ros_time"] for r in rows], dtype=np.float64)
        if n < 2 or np.any(np.diff(t) <= 0):
            return "bad_timestamps"

        dt = 1.0 / opts["fps"]
        t_grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-6)) + 1) * dt
        if len(t_grid) < opts["min_frames"]:
            return "too_short"
        idx = np.clip(np.searchsorted(t, t_grid), 1, n - 1)
        idx = np.where(np.abs(t_grid - t[idx - 1]) <= np.abs(t[idx] - t_grid), idx - 1, idx)
        if np.abs(t[idx] - t_grid).max() > opts["max_slop_ms"] / 1000:
            return "step_gap"

        pos = np.array([r["eef_pos"] for r in rows], dtype=np.float64)
        quat = np.array([r["eef_quat"] for r in rows], dtype=np.float64)
        width = np.array([r["gripper_width"] for r in rows], dtype=np.float64)
        to_base = Rotation.from_euler("z", -opts["base_yaw_deg"], degrees=True)
        xyz = to_base.apply(np.stack([np.interp(t_grid, t, pos[:, k]) for k in range(3)], axis=1))
        rot = to_base * Slerp(t, Rotation.from_quat(quat))(t_grid)
        cartesian = np.concatenate([xyz, rot.as_euler("xyz")], axis=1)
        closed = np.clip(1.0 - np.interp(t_grid, t, width) / opts["gripper_open"], 0.0, 1.0)

        h_out = opts["height"]
        views = {"front": [], "hand": []}
        x0, y0, x1, y1 = OVERLAY_BOX
        for i in idx:
            img = frames[i]
            half = img.shape[1] // 2
            front, hand = img[:, :half], img[:, half:]
            mask = np.zeros(front.shape[:2], np.uint8)
            mask[y0:y1, x0:x1] = 255
            front = cv2.inpaint(np.ascontiguousarray(front), mask, 5, cv2.INPAINT_TELEA)
            for name, view in (("front", front), ("hand", hand)):
                if h_out:
                    w_out = 2 * round(view.shape[1] * h_out / view.shape[0] / 2)
                    view = cv2.resize(view, (w_out, h_out), interpolation=cv2.INTER_AREA)
                views[name].append(np.ascontiguousarray(view))

        os.makedirs(out, exist_ok=True)
        for name, frames_v in views.items():
            write_video(frames_v, os.path.join(out, f"{name}.mp4"), opts["fps"], opts["crf"])
        meta = json.load(open(os.path.join(ep, "episode.json")))
        tmp = os.path.join(out, "data.h5.tmp")
        with h5py.File(tmp, "w") as f:
            for name in views:
                f.create_dataset(f"cameras/{name}", data=name.encode())
            f.create_dataset("trajectory/timestamp", data=t_grid)
            state = f.create_group("trajectory/observation/robot_state")
            state.create_dataset("cartesian_position", data=cartesian.astype(np.float32))
            state.create_dataset("gripper_position", data=closed.astype(np.float32))
            f.attrs.update(source=os.path.realpath(ep), fps=opts["fps"], base_yaw_deg=opts["base_yaw_deg"],
                           gripper_open=opts["gripper_open"], policy=str(meta.get("policy")),
                           task=str(meta.get("task")), phase=str(meta.get("phase")),
                           outcome=str(meta.get("outcome")))
        os.replace(tmp, os.path.join(out, "data.h5"))
        return None
    except Exception as e:  # a corrupt episode is dropped, not the run
        return f"error:{type(e).__name__}:{' '.join(str(e).split())[:200]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, nargs="+", help="benchmark dirs (<results_dir>/<benchmark>)")
    parser.add_argument("--dst", required=True, help="output root")
    parser.add_argument("--fps", type=float, default=5.0, help="rate of the output timeline (the control rate)")
    parser.add_argument("--height", type=int, default=360, help="output video height, 0 keeps the source size")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality (lower is better)")
    parser.add_argument("--base_yaw_deg", type=float, default=90.0, help="yaw of the robot base in the pose frame")
    parser.add_argument("--gripper_open", type=float, default=0.04, help="gripper width of the open hand")
    parser.add_argument("--max_slop_ms", type=float, default=60.0, help="largest step-to-timeline offset")
    parser.add_argument("--min_frames", type=int, default=24, help="shorter episodes are dropped")
    parser.add_argument("--overwrite", action="store_true", help="reconvert episodes that already have a data.h5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    opts = {k: getattr(args, k) for k in (
        "fps", "height", "crf", "base_yaw_deg", "gripper_open", "max_slop_ms", "min_frames", "overwrite")}

    dst = os.path.abspath(os.path.expanduser(args.dst))
    tasks = []
    for src in [os.path.abspath(os.path.expanduser(s)) for s in args.src]:
        if not os.path.isdir(src):
            continue
        episodes = find_episodes(src)
        print(f"{src}: {len(episodes)} episodes")
        tasks += [(ep, os.path.join(dst, os.path.basename(src), os.path.relpath(ep, src)), opts) for ep in episodes]

    kept, dropped = [], []
    with Pool(args.workers) as pool:
        for i, reason in enumerate(pool.imap(convert_episode, tasks, chunksize=1)):
            rel = os.path.relpath(tasks[i][1], dst)
            (dropped.append((rel, reason)) if reason else kept.append((rel, tasks[i][0])))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} done, {len(dropped)} dropped", flush=True)

    os.makedirs(dst, exist_ok=True)
    kept.sort()
    with open(os.path.join(dst, "dataset_index.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["path"]] + [[rel] for rel, _ in kept])
    with open(os.path.join(dst, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "benchmark", "policy", "task", "phase", "outcome"])
        for rel, ep in kept:
            meta = json.load(open(os.path.join(ep, "episode.json")))
            w.writerow([rel, rel.split("/")[0], meta.get("policy"), meta.get("task"), meta.get("phase"),
                        meta.get("outcome")])
    with open(os.path.join(dst, "dropped_episodes.tsv"), "w") as f:
        f.writelines(f"{rel}\t{reason}\n" for rel, reason in sorted(dropped))
    reasons = {}
    for _, r in dropped:
        reasons[r.split(":")[0]] = reasons.get(r.split(":")[0], 0) + 1
    print(f"kept {len(kept)} episodes, dropped {len(dropped)} {reasons}; index: {dst}/dataset_index.csv")


if __name__ == "__main__":
    main()
