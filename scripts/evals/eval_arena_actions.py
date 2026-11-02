# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Arena action-sensitivity eval: does the world model's rollout depend on the actions it is given?

Same clips, prefixes and rollouts as eval_arena_holdout.py (the clips are encoded with its encode_clips and the same
seed), but the inputs of the rollout steps are replaced. For every prefix t the history (real frames 0..t and the
teacher-forced prediction of frame t+1, made with the true action t) is shared; rollout step j (j = 1..9) predicts
frame t+1+j from the actions t+1..t+j and the poses t+2..t+j. Those actions and poses come from one of these modes:

  * true:          the clip's own actions; poses integrated from them (equal to the dataset poses up to rounding, so
                   this reproduces eval_arena_holdout.py and checks the integration).
  * shuffle:       actions of another clip (a different episode, same time indices); poses integrated from them.
  * zero:          zero actions; the arm pose stays at pose t+1 (the arm stops after the last real step).
  * shuffle_act:   another clip's actions, the true poses (only the action input is wrong).
  * shuffle_pose:  the true actions, poses integrated from another clip's actions (only the pose input is wrong).

Poses are integrated like RobotVideoDataset.poses_to_diffs in reverse: xyz += dxyz, R = R(d_euler) @ R, gripper
closedness += d (clipped to [0, 1] as in the converters).

Per mode and rollout step j, averaged over all (clip, prefix) samples:
  * l1_gt:     L1 to the real future features (the eval_arena_holdout metric). A model that uses the actions gets worse
               with wrong actions.
  * l1_true:   L1 to the prediction of the true mode (how much the prediction moves when the inputs change).
  * l1_static: L1 to the real features of frame t+1 (the arm's last real frame; with zero actions the scene should stay
               there).
  * win_true:  fraction of samples where the true-action rollout is closer to the real future than this mode's
               (0.5 = the model cannot tell the true actions from these).
Also reported: mean translation per step of the true and donor actions (m), so a shuffle is a real change of motion.

Outputs in --out-dir: <data>/<model>.json per pair (skipped when it exists) and summary.md / summary.json.

    PYTHONPATH=$PWD python scripts/evals/eval_arena_actions.py \\
        --data sim_heldout=/data/real1030_heldout:30 benchmark=/data/rs1030_benchmark:5 \\
        --model pretrained=/w/wm-100m/cooldown_100m_150k.pth.tar final=/runs/arena/world_model-final.pth.tar \\
        --config /w/wm-100m/config.json --encoder /w/vjepa2_1_vitG_384.pt --out-dir results/arena_100m_actions
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_arena_holdout as holdout  # noqa: E402
import eval_droid_holdout as droid  # noqa: E402

DTYPE = droid.DTYPE
STEPS = droid.EVAL_ROLLOUT_STEPS  # rollout steps 1..STEPS-1 are reported
MODES = ["true", "shuffle", "zero", "shuffle_act", "shuffle_pose"]
GRIPPER_IDX = holdout.POSE_DIM  # state/action layout: xyz(3) euler xyz(3) gripper(1), zero padded


def integrate(pose0, actions):
    """pose0 [B, D], actions [B, K, D] (float64 numpy) -> poses after each action [B, K, D]."""
    b, k, d = actions.shape
    out = np.zeros((b, k, d))
    pos = pose0[:, :3].copy()
    rot = Rotation.from_euler("xyz", pose0[:, 3:6])
    grip = pose0[:, GRIPPER_IDX].copy()
    for i in range(k):
        pos = pos + actions[:, i, :3]
        rot = Rotation.from_euler("xyz", actions[:, i, 3:6]) * rot
        grip = np.clip(grip + actions[:, i, GRIPPER_IDX], 0.0, 1.0)
        out[:, i, :3] = pos
        out[:, i, 3:6] = rot.as_euler("xyz")
        out[:, i, GRIPPER_IDX] = grip
    return out


def pose_error(a, b):
    """Max xyz (m), rotation angle (rad), gripper and raw euler differences between pose arrays [..., D]."""
    ra = Rotation.from_euler("xyz", a[..., 3:6].reshape(-1, 3))
    rb = Rotation.from_euler("xyz", b[..., 3:6].reshape(-1, 3))
    return {
        "xyz": float(np.abs(a[..., :3] - b[..., :3]).max()),
        "rot": float((ra * rb.inv()).magnitude().max()),
        "gripper": float(np.abs(a[..., GRIPPER_IDX] - b[..., GRIPPER_IDX]).max()),
        "euler": float(np.abs(a[..., 3:6] - b[..., 3:6]).max()),
    }


def assign_donors(clips, seed):
    """For every clip, the actions of another clip from a different episode with at least as many valid steps."""
    rng = np.random.default_rng(seed)
    valid = [int(c["padding_mask"].sum()) for c in clips]
    episode = [c["id"].rsplit("@", 1)[0] for c in clips]
    for i, clip in enumerate(clips):
        candidates = [j for j in range(len(clips)) if valid[j] >= valid[i] and episode[j] != episode[i]]
        j = int(rng.choice(candidates))
        clip["donor_action"] = clips[j]["action"].clone()
        clip["donor_id"] = clips[j]["id"]


def batches(clips, batch_size):
    """droid.batches plus the donor actions."""
    groups = defaultdict(list)
    for clip in clips:
        groups[int(clip["padding_mask"].sum())].append(clip)
    for _, group in sorted(groups.items()):
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            out = {k: torch.cat([c[k] for c in chunk]) for k in ("features", "action", "state", "extrinsics",
                                                                   "padding_mask", "view_list", "dt", "donor_action")}
            out["embodiment"] = [e for c in chunk for e in c["embodiment"]]
            yield out


def frame_l1(a, b):
    """Per-sample, per-frame L1 of features [B, T, V, H, W, C] -> [B, T] (compute_loss's mae without masking)."""
    return (a.float() - b.float()).abs().mean(dim=(2, 3, 4, 5))


def mode_inputs(mode, t, action, donor, state, n):
    """Actions t+1..t+n and poses t+2..t+n+1 (numpy float64) for one prefix."""
    true_act = action[:, t + 1:t + 1 + n]
    donor_act = donor[:, t + 1:t + 1 + n]
    pose0 = state[:, t + 1]
    true_pose = state[:, t + 2:t + 2 + n]
    if mode == "true":
        return true_act, integrate(pose0, true_act)
    if mode == "shuffle":
        return donor_act, integrate(pose0, donor_act)
    if mode == "zero":
        zero = np.zeros_like(true_act)
        return zero, integrate(pose0, zero)
    if mode == "shuffle_act":
        return donor_act, true_pose
    if mode == "shuffle_pose":
        return true_act, integrate(pose0, donor_act)
    raise ValueError(mode)


def eval_actions(world_model, clips, device, batch_size, modes):
    n = STEPS - 1  # rollout steps, each needs one new action and one new pose
    sums, cnts = defaultdict(float), defaultdict(int)
    integ_err = defaultdict(float)
    for clip in batches(clips, batch_size):
        to = lambda x: x.to(device, dtype=DTYPE)  # noqa: E731
        action, state, extrinsics, dt = to(clip["action"]), to(clip["state"]), to(clip["extrinsics"]), to(clip["dt"])
        padding_mask, view_list = to(clip["padding_mask"]), clip["view_list"].to(device)
        action_np = clip["action"].double().numpy()
        donor_np = clip["donor_action"].double().numpy()
        state_np = clip["state"].double().numpy()
        embodiment_ids = None
        if world_model.model.is_multiembodiment:
            embodiment_ids = torch.tensor([droid.embodiment_id(world_model, e) for e in clip["embodiment"]],
                                          device=device)
        video_f = clip["features"].to(device)
        if world_model.encoder_layernorm:
            video_f = F.layer_norm(video_f, (video_f.size(-1),))
        if video_f.shape[1] == action.shape[1] + 1:
            video_f, padding_mask = video_f[:, :-1], padding_mask[:, :-1]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            predicted, _, predicted_states, _ = world_model.forward(
                video_f, action, state, extrinsics, view_map=view_list, dt=dt, embodiment_ids=embodiment_ids)
            total_steps = int(padding_mask.sum(1).min().item())
            for t in range(total_steps + 1 - STEPS):
                history = world_model.select_history(
                    video_f, action, predicted, states=state, predicted_states=predicted_states,
                    camera_extrinsics=extrinsics, dt=dt, prefix_len=t + 1, suffix_len=1)
                target = video_f[:, t + 2:t + 2 + n]  # real frames t+2..t+1+n
                static = video_f[:, t + 1:t + 2]  # real frame t+1
                preds, l1_gt = {}, {}
                for mode in modes:
                    acts, poses = mode_inputs(mode, t, action_np, donor_np, state_np, n)
                    if mode == "true":
                        err = pose_error(poses, state_np[:, t + 2:t + 2 + n])
                        for k, v in err.items():
                            integ_err[k] = max(integ_err[k], v)
                    # rollout reads actions_suffix[:, h] and states_suffix[:, h] at step h
                    _, final = world_model.rollout(
                        history, actions_suffix=torch.from_numpy(acts).to(device, dtype=DTYPE), features_suffix=None,
                        states_suffix=torch.from_numpy(poses).to(device, dtype=DTYPE), dt=dt[:, t + 2:],
                        action_noise=0.0, suffix_len=1, run_backward=False, padding_mask=padding_mask[:, t + 1:-1],
                        rollout_steps=n, view_map=view_list, embodiment_ids=embodiment_ids)
                    pred = final.features[:, t + 2:t + 2 + n]
                    assert pred.shape[1] == n, pred.shape
                    preds[mode] = pred
                    l1_gt[mode] = frame_l1(pred, target)  # [B, n]
                    metrics = {"l1_gt": l1_gt[mode], "l1_static": frame_l1(pred, static.expand_as(pred))}
                    if mode != "true":
                        metrics["l1_true"] = frame_l1(pred, preds["true"])
                        metrics["win_true"] = (l1_gt["true"] < l1_gt[mode]).float()
                    for name, values in metrics.items():
                        for j in range(n):
                            sums[f"{mode}/{name}/{j + 1}"] += float(values[:, j].sum())
                            cnts[f"{mode}/{name}/{j + 1}"] += values.shape[0]
                        sums[f"{mode}/{name}/mean"] += float(values.mean(1).sum())
                        cnts[f"{mode}/{name}/mean"] += values.shape[0]
                    if mode != "true":  # per-sample decision over the whole 9-step rollout
                        win = (l1_gt["true"].mean(1) < l1_gt[mode].mean(1)).float()
                        sums[f"{mode}/win_true_rollout"] += float(win.sum())
                        cnts[f"{mode}/win_true_rollout"] += win.shape[0]
                for name, acts in (("true", action_np), ("donor", donor_np)):
                    step = np.linalg.norm(acts[:, t + 1:t + 1 + n, :3], axis=-1)  # [B, n] translation per step
                    sums[f"action_translation/{name}"] += float(step.mean(1).sum())
                    cnts[f"action_translation/{name}"] += step.shape[0]
    metrics = {k: sums[k] / cnts[k] for k in sorted(sums)}
    return metrics, dict(cnts), dict(integ_err)


def per_step(metrics, mode, name):
    return [metrics.get(f"{mode}/{name}/{j}", float("nan")) for j in range(1, STEPS)]


def write_summary(out_dir):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*", "*.json")))]
    by_data = defaultdict(dict)
    for r in rows:
        by_data[r["data"]][r["model"]] = r
    lines = ["# Arena action sensitivity", "",
             "Rollouts from the same history with the true, another episode's (shuffle) or zero actions; poses "
             "integrated from the actions. shuffle_act / shuffle_pose: only the action / only the pose input is "
             "wrong. l1_gt: L1 to the real future (mean over steps 1-9, step j = j+1 WM steps of 0.4 s past the last "
             "real frame). l1_true: L1 to the true-action prediction. l1_static: L1 to the last real frame. win: "
             "fraction of samples where the true-action rollout is closer to the real future (9-step mean; 0.5 = no "
             "preference).", ""]
    for data, models in sorted(by_data.items()):
        any_r = next(iter(models.values()))
        lines += [f"## {data} ({any_r['episodes']} episodes, {any_r['n_clips']} clips, translation per step: true "
                  f"{1000 * any_r['metrics']['action_translation/true']:.1f} mm, donor "
                  f"{1000 * any_r['metrics']['action_translation/donor']:.1f} mm)", "",
                  "| model | mode | l1_gt 1-9 | vs true | l1_gt step 1 | l1_gt step 9 | l1_true 1-9 | l1_static 1-9 "
                  "| win (rollout) |", "|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(models, key=lambda m: (m != "pretrained", models[m].get("epoch") or 0, m)):
            r = models[m]
            mt = r["metrics"]
            base = mt["true/l1_gt/mean"]
            for mode in r["modes"]:
                gt = per_step(mt, mode, "l1_gt")
                rel = l1_true = win = ""
                if mode != "true":
                    rel = f"{100 * (mt[mode + '/l1_gt/mean'] / base - 1):+.1f}%"
                    l1_true = f"{mt[mode + '/l1_true/mean']:.4f}"
                    win = f"{mt[mode + '/win_true_rollout']:.3f}"
                lines.append(f"| {m} | {mode} | {mt[mode + '/l1_gt/mean']:.4f} | {rel} | {gt[0]:.4f} | {gt[-1]:.4f} | "
                             f"{l1_true} | {mt[mode + '/l1_static/mean']:.4f} | {win} |")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({d: {m: {"epoch": r["epoch"], "integration_error": r["integration_error"],
                           **{mode: {name: per_step(r["metrics"], mode, name)
                                     for name in ("l1_gt", "l1_true", "l1_static", "win_true")}
                              for mode in r["modes"]}} for m, r in ms.items()} for d, ms in by_data.items()},
                  f, indent=2)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, nargs="+", help="NAME=PATH:RAW_FPS (converted Arena root)")
    parser.add_argument("--model", required=True, nargs="+", help="NAME=CHECKPOINT (.pth.tar)")
    parser.add_argument("--config", required=True, help="transition-model config.json of the checkpoints")
    parser.add_argument("--encoder", required=True, help="V-JEPA 2.1 ViT-G/384 weights (vjepa2_1_vitG_384.pt)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--n-clips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16, help="clips per forward (same valid length)")
    args = parser.parse_args()
    modes = ["true"] + [m for m in args.modes if m != "true"]  # the other modes are compared with true
    device = "cuda"
    torch.cuda.set_device(0)

    data = {}
    for name, value in holdout.parse_pairs(args.data, "data").items():
        path, _, fps = value.rpartition(":")
        data[name] = (path, float(fps))
    models = holdout.parse_pairs(args.model, "model")

    encoder = None
    for dname, (path, raw_fps) in data.items():
        out = os.path.join(args.out_dir, dname)
        os.makedirs(out, exist_ok=True)
        todo = [m for m in models if not os.path.exists(os.path.join(out, f"{m}.json"))]
        print(f"== {dname}: {len(todo)} of {len(models)} to evaluate", flush=True)
        if not todo:
            continue
        clips = n_episodes = None
        for mname in todo:
            t0 = time.time()
            wm, epoch = holdout.load_model(models[mname], args.config, args.encoder, encoder, device)
            encoder = wm.encoder
            if clips is None:
                clips, n_episodes = holdout.encode_clips(encoder, dname, path, raw_fps, args.n_clips, args.seed,
                                                         device)
                assign_donors(clips, args.seed)
            droid.seed_all(args.seed)
            metrics, counts, integ_err = eval_actions(wm, clips, device, args.batch_size, modes)
            result = {
                "model": mname, "checkpoint": models[mname], "epoch": epoch, "data": dname, "data_path": path,
                "episodes": n_episodes, "n_clips": len(clips), "seed": args.seed, "batch_size": args.batch_size,
                "modes": modes, "integration_error": integ_err, "metrics": metrics, "counts": counts,
                "clip_ids_sha": hashlib.sha256("\n".join(c["id"] for c in clips).encode()).hexdigest(),
                "donor_ids_sha": hashlib.sha256("\n".join(c["donor_id"] for c in clips).encode()).hexdigest(),
                "seconds": time.time() - t0,
            }
            with open(os.path.join(out, f"{mname}.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"{dname} / {mname} ({time.time() - t0:.0f}s), integration error {integ_err}", flush=True)
            for mode in modes:
                print(f"  {mode:13s} l1_gt {metrics[f'{mode}/l1_gt/mean']:.4f}  l1_static "
                      f"{metrics[f'{mode}/l1_static/mean']:.4f}  win "
                      f"{metrics.get(f'{mode}/win_true_rollout', float('nan')):.3f}", flush=True)
            del wm
            torch.cuda.empty_cache()
        del clips
    write_summary(args.out_dir)


if __name__ == "__main__":
    main()
