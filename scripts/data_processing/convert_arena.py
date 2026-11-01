#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert raw Arena episodes (Franka, sim or real) into the episode format the robo_jepa dataloader reads.

Input: trees of raw episode directories as written by the Arena data collection (the input of
``openpi/data_processing/couple_images_eef_poses_adj_freq.py``)::

    <episode>/
        <camera>_color.mp4, <camera>_rgb_timestamps.npy   # camera in front / left / right / hand
        eef_poses.npy                                      # [N, 8]: t, x, y, z, qx, qy, qz, qw (world frame)
        gripper_positions.npy                              # [N, 2]: t, opening (largest = open)
        joint_states.npy                                   # [N, 8]: t, 7 joint angles (optional)
        quality_review.json                                # optional; needs_review=true episodes are dropped

Every directory under a ``--src`` root that holds an ``eef_poses.npy`` is an episode (symlinked directories are
followed, and an episode reachable through several paths is converted once).

Output, under ``--dst``::

    dataset_index.csv                       # column "path": episode dirs relative to --dst
    dropped_episodes.tsv                    # episode <TAB> reason, for every episode left out
    <basename of --src>/<episode path relative to --src>/
        data.h5
        <camera>.mp4                        # H.264, resampled to the uniform --fps timeline (and resized)

``data.h5`` follows the DROID (``franka`` embodiment) conventions of ``app/robo_jepa/docs/action_spaces/droid.md``::

    cameras/<camera>                                      # "<camera>" (the loader appends ".mp4")
    trajectory/timestamp                                  # [T] time of each frame on the source clock (s)
    trajectory/observation/robot_state/
        cartesian_position  [T, 6]                        # EE xyz (m) + extrinsic euler xyz (rad), robot base frame
        gripper_position    [T]                           # 0 = open, 1 = closed
        joint_positions     [T, 7]                        # rad (when joint_states.npy exists)

Row ``i`` of every array belongs to frame ``i`` of every video, which is what the loader assumes. The source streams
are not aligned that way (cameras drop frames and have their own clocks, the pose stream starts a tick after the
first frame), so each episode is resampled to a uniform ``--fps`` timeline over the span all streams cover: each
camera contributes its nearest frame (the episode is dropped if one is more than ``--max_slop_ms`` away), positions,
joints and the gripper are interpolated linearly and rotations by slerp.

Poses are rotated from the Arena world frame, where the robot base is yawed by ``--base_yaw_deg`` (``base_ori`` in
the collection config: 90, the robot faces +y), into the robot base frame DROID uses (x forward, y left, z up), so a
top-down grasp has roll near +-pi as in DROID. The gripper opening g becomes the closedness ``clip(1 - g / g_open, 0,
1)``, with ``g_open`` (per ``--src`` root) the median of the episodes' largest openings (sim 0.5, real ~0.99) unless
``--gripper_open`` is given.

    python scripts/data_processing/convert_arena.py --src ~/arena_sim_data/realmatch_dr_multi \\
        --dst ~/arena_wm_data --workers 16

Configs then point ``data.s3_prefix`` at ``--dst`` with ``raw_fps`` equal to ``--fps`` and ``embodiment_name:
franka`` (see ``configs/finetuning/arena_100m.yaml``). Already converted episodes are skipped unless ``--overwrite``,
so an interrupted run can be restarted; ``dataset_index.csv`` lists the episodes of the latest run's ``--src`` roots.
"""

import argparse
import csv
import json
import os
from fractions import Fraction
from multiprocessing import Pool

import av
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def find_episodes(src):
    """Episode dirs under ``src``, relative to it. Each real directory is visited once, in sorted order."""
    seen, episodes = set(), []
    for d, dirs, files in os.walk(src, followlinks=True):
        dirs.sort()
        real = os.path.realpath(d)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        if "eef_poses.npy" in files:
            episodes.append(os.path.relpath(d, src))
            dirs[:] = []
    return episodes


def gripper_open(src, episodes):
    """Median over the episodes of their largest gripper opening."""
    maxima = []
    for rel in episodes:
        path = os.path.join(src, rel, "gripper_positions.npy")
        if os.path.exists(path):
            g = np.load(path)[:, 1]
            if np.isfinite(g).any():
                maxima.append(np.nanmax(g))
    return float(np.median(maxima))


def load_series(path):
    """(t, values) of a [N, 1 + D] array: rows with NaNs dropped, sorted by time, duplicate times dropped."""
    a = np.load(path).astype(np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    t, idx = np.unique(a[:, 0], return_index=True)
    return t, a[idx, 1:]


def nearest(t_src, t):
    i = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
    return np.where(np.abs(t - t_src[i - 1]) <= np.abs(t_src[i] - t), i - 1, i)


def interp(t_src, values, t):
    return np.stack([np.interp(t, t_src, values[:, k]) for k in range(values.shape[1])], axis=1)


def open_video(path):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = 2
    return container, stream


def resample_video(src, timestamps, t_grid, dst, fps, height, crf):
    """Write the frames of ``src`` nearest to ``t_grid`` to ``dst``. ``timestamps[i]`` is the time of frame i."""
    idx = nearest(timestamps, t_grid)
    container, stream = open_video(src)
    rate = Fraction(fps).limit_denominator(1001)
    n = 0
    with container, av.open(dst, "w") as out:
        w, h = stream.codec_context.width, stream.codec_context.height
        if height:
            w, h = 2 * round(w * height / h / 2), height
        enc = out.add_stream("libx264", rate=rate)
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.options = {"crf": str(crf), "preset": "veryfast", "g": "30"}
        for i, frame in enumerate(container.decode(stream)):
            if n == len(idx):
                break
            if idx[n] != i:
                continue
            frame = frame.reformat(width=w, height=h, format="yuv420p", interpolation="AREA")
            while n < len(idx) and idx[n] == i:  # a frame the source dropped repeats its nearest neighbour
                frame.pts, frame.time_base = n, 1 / rate
                for packet in enc.encode(frame):
                    out.mux(packet)
                n += 1
        for packet in enc.encode():
            out.mux(packet)
    if n != len(t_grid):
        raise ValueError(f"{os.path.basename(src)} has {n} of the {len(t_grid)} frames needed")


def convert_episode(args):
    ep, out, g_open, opts = args
    if not opts["overwrite"] and os.path.exists(os.path.join(out, "data.h5")):
        return None
    try:
        review = os.path.join(ep, "quality_review.json")
        if not opts["keep_needs_review"] and os.path.exists(review):
            with open(review) as f:
                if json.load(f).get("needs_review"):
                    return "needs_review"

        cameras = {}
        for cam in opts["cameras"]:
            mp4, ts = os.path.join(ep, f"{cam}_color.mp4"), os.path.join(ep, f"{cam}_rgb_timestamps.npy")
            if not (os.path.exists(mp4) and os.path.exists(ts)):
                return f"missing_camera:{cam}"
            container, stream = open_video(mp4)
            with container:
                n_frames = stream.frames
            ts = np.load(ts).astype(np.float64)
            cameras[cam] = (mp4, ts[: n_frames or len(ts)])
            if len(cameras[cam][1]) < 2 or np.any(np.diff(cameras[cam][1]) <= 0):
                return f"bad_timestamps:{cam}"

        t_pose, pose = load_series(os.path.join(ep, "eef_poses.npy"))
        valid = np.linalg.norm(pose[:, 3:7], axis=1) > 1e-6
        t_pose, pose = t_pose[valid], pose[valid]
        t_grip, grip = load_series(os.path.join(ep, "gripper_positions.npy"))
        joints_path = os.path.join(ep, "joint_states.npy")
        t_joint, joints = load_series(joints_path) if os.path.exists(joints_path) else (None, None)
        state_ts = [t_pose, t_grip] + ([t_joint] if t_joint is not None else [])
        if min(len(t) for t in state_ts) < 2:
            return "empty_state"

        # Uniform timeline inside every camera's span. State streams may end up to one frame early or start one
        # frame late (they are held constant there): the pose stream starts one tick after the first frame.
        dt = 1.0 / opts["fps"]
        lo = max([ts[0] for _, ts in cameras.values()] + [t[0] - dt for t in state_ts])
        hi = min([ts[-1] for _, ts in cameras.values()] + [t[-1] + dt for t in state_ts])
        t_grid = lo + np.arange(int(np.floor((hi - lo) / dt + 1e-6)) + 1) * dt
        if len(t_grid) < opts["min_frames"]:
            return "too_short"
        for cam, (_, ts) in cameras.items():
            if np.abs(ts[nearest(ts, t_grid)] - t_grid).max() > opts["max_slop_ms"] / 1000:
                return f"camera_gap:{cam}"

        to_base = Rotation.from_euler("z", -opts["base_yaw_deg"], degrees=True)
        xyz = to_base.apply(interp(t_pose, pose[:, :3], t_grid))
        rot = to_base * Slerp(t_pose, Rotation.from_quat(pose[:, 3:7]))(np.clip(t_grid, t_pose[0], t_pose[-1]))
        cartesian = np.concatenate([xyz, rot.as_euler("xyz")], axis=1)
        closed = np.clip(1.0 - interp(t_grip, grip[:, :1], t_grid)[:, 0] / g_open, 0.0, 1.0)

        os.makedirs(out, exist_ok=True)
        for cam, (mp4, ts) in cameras.items():
            resample_video(mp4, ts, t_grid, os.path.join(out, f"{cam}.mp4"), opts["fps"], opts["height"], opts["crf"])
        tmp = os.path.join(out, "data.h5.tmp")
        with h5py.File(tmp, "w") as f:
            for cam in cameras:
                f.create_dataset(f"cameras/{cam}", data=cam.encode())
            f.create_dataset("trajectory/timestamp", data=t_grid)
            state = f.create_group("trajectory/observation/robot_state")
            state.create_dataset("cartesian_position", data=cartesian.astype(np.float32))
            state.create_dataset("gripper_position", data=closed.astype(np.float32))
            if joints is not None:
                state.create_dataset("joint_positions", data=interp(t_joint, joints, t_grid).astype(np.float32))
            f.attrs.update(source=os.path.realpath(ep), fps=opts["fps"], base_yaw_deg=opts["base_yaw_deg"],
                           gripper_open=g_open)
        os.replace(tmp, os.path.join(out, "data.h5"))
        return None
    except Exception as e:  # a corrupt episode is dropped, not the run
        return f"error:{type(e).__name__}:{' '.join(str(e).split())[:200]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, nargs="+", help="roots holding raw episode dirs")
    parser.add_argument("--dst", required=True, help="output root")
    parser.add_argument("--cameras", nargs="+", default=["front", "hand"], help="cameras every episode must have")
    parser.add_argument("--fps", type=float, default=30.0, help="rate of the output timeline (the config's raw_fps)")
    parser.add_argument("--height", type=int, default=360, help="output video height, 0 keeps the source size")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality (lower is better)")
    parser.add_argument("--base_yaw_deg", type=float, default=90.0, help="yaw of the robot base in the pose frame")
    parser.add_argument("--gripper_open", type=float, default=None, help="opening of the open gripper (default: auto)")
    parser.add_argument("--max_slop_ms", type=float, default=50.0, help="largest camera-to-timeline offset")
    parser.add_argument("--min_frames", type=int, default=30, help="shorter episodes are dropped")
    parser.add_argument("--keep_needs_review", action="store_true", help="keep episodes flagged in quality_review")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N episodes of each root")
    parser.add_argument("--overwrite", action="store_true", help="reconvert episodes that already have a data.h5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    opts = {k: getattr(args, k) for k in (
        "cameras", "fps", "height", "crf", "base_yaw_deg", "max_slop_ms", "min_frames", "keep_needs_review",
        "overwrite")}

    srcs = [os.path.abspath(os.path.expanduser(s)) for s in args.src]
    dst = os.path.abspath(os.path.expanduser(args.dst))
    names = [os.path.basename(s) for s in srcs]
    if len(set(names)) != len(names):
        raise SystemExit(f"--src roots must have distinct basenames, got {names}")
    tasks = []
    for src, name in zip(srcs, names):
        episodes = find_episodes(src)[: args.limit]
        g_open = args.gripper_open if args.gripper_open is not None else gripper_open(src, episodes)
        print(f"{src}: {len(episodes)} episodes, gripper open at {g_open:.4f}")
        tasks += [(os.path.join(src, rel), os.path.join(dst, name, rel), g_open, opts) for rel in episodes]

    kept, dropped = [], []
    with Pool(args.workers) as pool:
        for i, reason in enumerate(pool.imap(convert_episode, tasks, chunksize=1)):
            rel = os.path.relpath(tasks[i][1], dst)
            (dropped.append((rel, reason)) if reason else kept.append(rel))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)} done, {len(dropped)} dropped", flush=True)

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "dataset_index.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["path"]] + [[rel] for rel in sorted(kept)])
    with open(os.path.join(dst, "dropped_episodes.tsv"), "w") as f:
        f.writelines(f"{rel}\t{reason}\n" for rel, reason in sorted(dropped))
    reasons = {}
    for _, r in dropped:
        reasons[r.split(":")[0]] = reasons.get(r.split(":")[0], 0) + 1
    print(f"kept {len(kept)} episodes, dropped {len(dropped)} {reasons}; index: {dst}/dataset_index.csv")


if __name__ == "__main__":
    main()
