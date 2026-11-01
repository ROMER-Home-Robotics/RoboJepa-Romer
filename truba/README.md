# robo_jepa on TRUBA

Training runs inside an Apptainer image (`robo_jepa.sif`). The repository is bind-mounted at run time, so code and
configs change without a rebuild, as with `pi0.sif`.

| What | Where on TRUBA |
| --- | --- |
| Repository | `/arf/home/yyesilnur/Workspace/wm/RoboJepa-Romer` |
| Container | `/arf/scratch/yyesilnur/Workspace/containers/robo_jepa.sif` |
| Weights (`ROBO_JEPA_WEIGHTS`) | `/arf/scratch/yyesilnur/Workspace/parameters/robo_jepa/` (`vjepa2_1_vitG_384.pt`, `wm-100m/`) |
| Data (`ROBO_JEPA_DATA`) | `/arf/scratch/yyesilnur/Workspace/datasets_processed/robo_jepa/` (`real1030c/`) |
| Runs (`ROBO_JEPA_RUNS`) | `/arf/scratch/yyesilnur/Workspace/parameters/robo_jepa/runs/` |

## Files

- `robo_jepa.def`: the image. Python 3.11 + the pins of `environment.yml`, with torch 2.7.1 **cu128** (sm_90 for
  H100/H200 and sm_120 for the RTX PRO 6000). robocasa/robosuite are left out. The V-JEPA 2.1 code is baked in at
  `/opt/vjepa2` (`VJEPA2_HUB_REPO`) because compute nodes cannot reach GitHub for `torch.hub`.
- `robo_jepa_train.slurm`: the training job (one rank per allocated GPU).
- `build_robo_jepa.slurm`: builds the image on TRUBA instead of on a workstation.

## Build the image

On the workstation, then copy it over:

```bash
apptainer build --fakeroot robo_jepa.sif truba/robo_jepa.def
rsync -a --partial robo_jepa.sif truba:/arf/scratch/yyesilnur/Workspace/containers/
```

or on TRUBA: `cd <repo> && sbatch truba/build_robo_jepa.slurm`.

## Data

Arena episodes are converted on the workstation and only the converted copy goes to TRUBA:

```bash
R=~/arena_sim_data/real1030c_data
python scripts/data_processing/convert_arena.py --dst /some/tmp/real1030c --workers 10 \
    --src $R/offset_start $R/packed $R/pick_from_place $R/place_occupied $R/recovery $R/table_height $R/wrong_place_start
rsync -a /some/tmp/real1030c truba:/arf/scratch/yyesilnur/Workspace/datasets_processed/robo_jepa/
```

The configs read `${ROBO_JEPA_DATA}/<dataset s3_prefix>/dataset_index.csv`.

## Train

```bash
cd /arf/home/yyesilnur/Workspace/wm/RoboJepa-Romer
DRY_RUN=1 bash truba/robo_jepa_train.slurm arena_100m_real1030c    # checks every path, no GPU needed
sbatch truba/robo_jepa_train.slurm arena_100m_real1030c
sbatch --gres=gpu:4 -c 32 truba/robo_jepa_train.slurm arena_100m_real1030c   # 4 ranks, 4x the global batch
```

- The argument is a config path or a name under `configs/finetuning/`. `${VAR}` references in a config are expanded
  from the environment when `app/main.py` loads it.
- Checkpoints, logs and the wandb run go to the config's `folder`. Resubmitting the same config resumes from
  `<folder>/world_model-latest.pth.tar`; to start over, change `folder` or delete it.
- W&B logs online when `WANDB_API_KEY` is exported before `sbatch` or `~/.netrc` has a wandb login. Otherwise it logs
  offline into the run folder (`wandb sync <folder>/wandb/offline-run-*`). No key is stored in the repository.
- Job logs: `train-slurm-robo_jepa_train-<jobid>.{out,err}` in the submit directory.
