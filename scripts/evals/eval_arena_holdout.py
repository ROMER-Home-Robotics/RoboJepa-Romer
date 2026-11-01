# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Arena held-out forward-prediction eval: per-step rollout L1 of world-model checkpoints on converted Arena episodes,
next to a copy-last-frame baseline.

Same metric and rollout code as eval_droid_holdout.py (imported from it): latent rollouts from every prefix of each
clip, L1 against the frozen-encoder features, reported per rollout step 1..9. Rollout step j predicts frame t+1+j from
real frames 0..t (the first predicted frame, t+1, comes from the teacher-forced pass), so step j is j+1 WM steps
(0.4 s each at fps 2.5) ahead of the last real frame. Like the training rollouts, every step gets the dataset's true
future robot pose as its state token; the actions are the dataset's (true) actions.

  * copy_last: predicts every future frame as the last real frame t (step j: L1(z_t, z_{t+1+j}); teacher forcing:
    L1(z_t, z_{t+1})). A model has to beat it to be predicting change at all.
  * every --data set is encoded once (fixed seed, same clips for every model) and evaluated with every --model.
  * --model accepts released checkpoints (cooldown_*.pth.tar) and training checkpoints (world_model-*.pth.tar,
    optimizer state is dropped); --config is the transition-model config.json they share.

Outputs in --out-dir: <data>/<model>.json per pair (skipped when it exists, so reruns only add new models/sets), and
summary.md / summary.json over everything in --out-dir.

    PYTHONPATH=$PWD python scripts/evals/eval_arena_holdout.py \\
        --data sim_heldout=/data/real1030_heldout:30 benchmark=/data/rs1030_benchmark:5 \\
        --model pretrained=/w/wm-100m/cooldown_100m_150k.pth.tar e100=/runs/arena/world_model-e100.pth.tar \\
        --config /w/wm-100m/config.json --encoder /w/vjepa2_1_vitG_384.pt --out-dir results/arena_100m
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_droid_holdout as droid  # noqa: E402

from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset  # noqa: E402
from app.robo_jepa.transforms import make_transforms  # noqa: E402
from app.robo_jepa.world_model import WorldModel  # noqa: E402

DTYPE = droid.DTYPE
STEPS = droid.EVAL_ROLLOUT_STEPS  # 10: rollout steps 1..9 are reported

# data.train.Arena of configs/finetuning/arena_100m*.yaml
EMBODIMENT = "franka"
FPS = 2.5
CAMERAS = ["front", "hand"]
VIEW_MAP = {"front": 0, "hand": 2}
POSE_DIM, GRIPPER_DIM, JOINTS_DIM = 6, 1, 7
PAD_SIZE, FRAMES_PER_CLIP, MIN_LENGTH_TO_DROP = 64, 16, 11
EVAL_AUGS = droid.EVAL_AUGS


def parse_pairs(items, what):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        out[name] = value
    return out


def encode_clips(encoder, name, path, raw_fps, n_clips, seed, device):
    dataset = RobotVideoDataset(
        dataset_name=name, data_path=None, cameras=CAMERAS, view_map=VIEW_MAP, embodiment_name=EMBODIMENT,
        raw_fps=raw_fps, fps=FPS, pose_dim=POSE_DIM, gripper_dim=GRIPPER_DIM, joints_dim=JOINTS_DIM, s3_profile=None,
        s3_prefix=path.rstrip("/") + "/", pad_size=PAD_SIZE, frames_per_clip=FRAMES_PER_CLIP,
        min_length_to_drop=MIN_LENGTH_TO_DROP, min_size=n_clips, transform=make_transforms(**EVAL_AUGS),
        read_intermediate_frames=False, use_joint_state=False,
    )
    n_episodes = len(dataset.samples)
    print(f"{name}: {n_episodes} episodes", flush=True)
    droid.seed_all(seed)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n_clips].tolist()
    clips, t0 = [], time.time()
    for i, idx in enumerate(order):
        (rgb_video, _, action, state, extrinsics, padding_mask, view_list, dt, start, traj_names, emb_names,
         index) = dataset[idx]
        video = rgb_video.unsqueeze(0).to(device, dtype=DTYPE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            b, _, v, t = video.shape[:4]
            h_, w_ = (s // encoder.vit.patch_size for s in video.shape[-2:])
            flat = einops.repeat(einops.rearrange(video, "b c v t h w -> (b t v) c h w"), "b c h w -> b c t h w", t=1)
            feats = einops.rearrange(encoder(flat.float()), "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_,
                                     w=w_)
        clips.append({
            "features": feats.to(DTYPE).cpu(), "action": action.unsqueeze(0), "state": state.unsqueeze(0),
            "extrinsics": extrinsics.unsqueeze(0), "padding_mask": padding_mask.unsqueeze(0),
            "view_list": view_list.unsqueeze(0), "dt": dt.unsqueeze(0),
            "embodiment": [emb_names] if isinstance(emb_names, str) else list(emb_names),
            "id": f"{traj_names}@{int(start)}",
        })
        if (i + 1) % 128 == 0:
            print(f"  encoded {i + 1}/{len(order)} clips ({time.time() - t0:.0f}s)", flush=True)
    return clips, n_episodes


def load_model(ckpt, config_path, encoder_path, encoder, device):
    """WorldModel.load_pretrained on a temp dir: config.json (attn_impl sdpa), the predictor weights of `ckpt` (released
    or training checkpoint), the encoder. load_pretrained loads non-strictly, so check that every weight was found."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = state.get("predictor") or state["model"]
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    tmp = tempfile.mkdtemp(prefix="wm_arena_")
    try:
        with open(config_path) as f:
            config = json.load(f)
        config["attn_impl"] = "sdpa"
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        torch.save({"model": weights}, os.path.join(tmp, "world_model.pth.tar"))
        os.symlink(os.path.abspath(encoder_path), os.path.join(tmp, "encoder.pth.tar"))
        wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=encoder is None, device=device)
    finally:
        shutil.rmtree(tmp)
    missing, unexpected = wm.transition_model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}: missing {missing[:5]}..., unexpected {unexpected[:5]}...")
    if encoder is not None:
        wm.encoder = encoder
    wm.eval()
    wm.predictor_window_steps = 1
    wm.eval_rollout_steps = STEPS
    wm.rollout_steps = 2
    wm.rollout_stop_gradient = True
    return wm, state.get("epoch")


def copy_last(world_model, clips, device, batch_size):
    """The baseline with the eval's clips, prefixes, masks and loss: every future frame predicted as frame t."""
    sums, cnts = defaultdict(float), defaultdict(int)
    for clip in droid.batches(clips, batch_size):
        n = clip["features"].shape[0]
        padding_mask = clip["padding_mask"].to(device, dtype=DTYPE)
        view_list = clip["view_list"].to(device)
        video_f = clip["features"].to(device)
        if world_model.encoder_layernorm:
            video_f = F.layer_norm(video_f, (video_f.size(-1),))
        if video_f.shape[1] == clip["action"].shape[1] + 1:
            video_f, padding_mask = video_f[:, :-1], padding_mask[:, :-1]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            tf = world_model.compute_loss(video_f, video_f, None, None, padding_mask, shift=1, view_map=view_list)
            sums["teacher_forcing/l1_loss"] += float(tf["l1_loss"]) * n
            cnts["teacher_forcing/l1_loss"] += n
            total_steps = int(padding_mask.sum(1).min().item())
            per_step = defaultdict(list)
            for t in range(total_steps + 1 - STEPS):
                for j in range(1, STEPS):
                    losses = world_model.compute_loss(
                        video_f[:, t:t + 1], video_f[:, t + 1 + j:t + 2 + j], None, None,
                        padding_mask[:, t + j:t + j + 1], shift=0, view_map=view_list)
                    per_step[j].append(float(losses["l1_loss"]))
            for j, values in per_step.items():
                sums[f"rollout/l1_loss/{j}"] += float(np.mean(values)) * n
                cnts[f"rollout/l1_loss/{j}"] += n
    return {k: sums[k] / cnts[k] for k in sorted(sums)}, dict(cnts)


def result_dict(model, ckpt, epoch, data, path, n_episodes, clips, args, metrics, counts, seconds):
    per_step = [metrics.get(f"rollout/l1_loss/{i}", float("nan")) for i in range(1, STEPS)]
    return {
        "model": model, "checkpoint": ckpt, "epoch": epoch, "data": data, "data_path": path,
        "episodes": n_episodes, "n_clips": len(clips), "seed": args.seed, "batch_size": args.batch_size,
        "kv_cache": args.kv_cache, "teacher_forcing_l1": metrics.get("teacher_forcing/l1_loss", float("nan")),
        "rollout_l1_per_step": per_step, "rollout_l1_mean_1_9": float(np.mean(per_step)),
        "step9_rollout_l1": per_step[-1], "metrics": metrics, "counts": counts,
        "clip_ids_sha": hashlib.sha256("\n".join(c["id"] for c in clips).encode()).hexdigest(), "seconds": seconds,
    }


def write_summary(out_dir):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*", "*.json")))]
    by_data = defaultdict(dict)
    for r in rows:
        by_data[r["data"]][r["model"]] = r
    lines = ["# Arena held-out forward prediction", "",
             "Rollout L1 against frozen-encoder features (lower is better). Step j is j+1 WM steps (0.4 s each) "
             "past the last real frame; teacher forcing is 1 step. vs copy_last / vs pretrained: relative change of "
             "the 1-9 mean.", ""]
    for data, models in sorted(by_data.items()):
        base = models.get("copy_last")
        pre = models.get("pretrained")
        any_r = next(iter(models.values()))
        lines += [f"## {data} ({any_r['episodes']} episodes, {any_r['n_clips']} clips)", "",
                  "| model | epoch | teacher forcing | step 1 | step 3 | step 5 | step 9 | mean 1-9 | vs copy_last "
                  "| vs pretrained |", "|---|---|---|---|---|---|---|---|---|---|"]
        order = sorted(models, key=lambda m: (m != "copy_last", m != "pretrained", models[m].get("epoch") or 0, m))
        for m in order:
            r = models[m]
            s = r["rollout_l1_per_step"]
            rel = lambda ref: f"{100 * (r['rollout_l1_mean_1_9'] / ref['rollout_l1_mean_1_9'] - 1):+.1f}%" \
                if ref and ref is not r else ""  # noqa: E731
            lines.append(f"| {m} | {r.get('epoch') if r.get('epoch') is not None else ''} | "
                         f"{r['teacher_forcing_l1']:.4f} | {s[0]:.4f} | {s[2]:.4f} | {s[4]:.4f} | {s[8]:.4f} | "
                         f"{r['rollout_l1_mean_1_9']:.4f} | {rel(base)} | {rel(pre)} |")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({d: {m: {k: r[k] for k in ("epoch", "teacher_forcing_l1", "rollout_l1_per_step",
                                               "rollout_l1_mean_1_9", "episodes", "n_clips")}
                       for m, r in ms.items()} for d, ms in by_data.items()}, f, indent=2)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, nargs="+", help="NAME=PATH:RAW_FPS (converted Arena root)")
    parser.add_argument("--model", required=True, nargs="+", help="NAME=CHECKPOINT (.pth.tar)")
    parser.add_argument("--config", required=True, help="transition-model config.json of the checkpoints")
    parser.add_argument("--encoder", required=True, help="V-JEPA 2.1 ViT-G/384 weights (vjepa2_1_vitG_384.pt)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-clips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16, help="clips per forward (same valid length)")
    parser.add_argument("--kv-cache", action="store_true", help="roll out through the KV cache (WorldModel.step)")
    args = parser.parse_args()
    device = "cuda"
    torch.cuda.set_device(0)

    data = {}
    for name, value in parse_pairs(args.data, "data").items():
        path, _, fps = value.rpartition(":")
        data[name] = (path, float(fps))
    models = parse_pairs(args.model, "model")
    for name in ("copy_last",):
        if name in models:
            raise SystemExit(f"model name {name!r} is reserved")

    encoder = None
    for dname, (path, raw_fps) in data.items():
        out = os.path.join(args.out_dir, dname)
        os.makedirs(out, exist_ok=True)
        todo = [m for m in list(models) + ["copy_last"] if not os.path.exists(os.path.join(out, f"{m}.json"))]
        print(f"== {dname}: {len(todo)} of {len(models) + 1} to evaluate", flush=True)
        if not todo:
            continue
        clips = n_episodes = None
        for mname in todo:
            t0 = time.time()
            ckpt = models.get(mname, next(iter(models.values())))  # copy_last only borrows compute_loss
            wm, epoch = load_model(ckpt, args.config, args.encoder, encoder, device)
            encoder = wm.encoder
            if clips is None:
                clips, n_episodes = encode_clips(encoder, dname, path, raw_fps, args.n_clips, args.seed, device)
            droid.seed_all(args.seed)
            if mname == "copy_last":
                metrics, counts = copy_last(wm, clips, device, args.batch_size)
                ckpt, epoch = None, None
            else:
                metrics, counts = droid.eval_checkpoint(wm, clips, device, args.batch_size, args.kv_cache)
            result = result_dict(mname, ckpt, epoch, dname, path, n_episodes, clips, args, metrics, counts,
                                 time.time() - t0)
            with open(os.path.join(out, f"{mname}.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"{dname} / {mname}: teacher forcing {result['teacher_forcing_l1']:.4f}, step-1 "
                  f"{result['rollout_l1_per_step'][0]:.4f}, step-9 {result['step9_rollout_l1']:.4f}, mean 1-9 "
                  f"{result['rollout_l1_mean_1_9']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            del wm
            torch.cuda.empty_cache()
        del clips
    write_summary(args.out_dir)


if __name__ == "__main__":
    main()
