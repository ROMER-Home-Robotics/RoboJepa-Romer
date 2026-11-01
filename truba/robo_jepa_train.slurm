#!/bin/bash
#SBATCH --output=train-slurm-%x-%j.out
#SBATCH --error=train-slurm-%x-%j.err
#SBATCH --time=2-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=kolyoz9,kolyoz13,kolyoz19,kolyoz24,kolyoz30,kolyoz31,kolyoz60
#SBATCH -p kolyoz-cuda
#SBATCH -A romer
#SBATCH --qos=midprio
#SBATCH -J robo_jepa_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -C "H100|H200"

# ==============================================================================
# robo_jepa world-model training on TRUBA, inside the robo_jepa.sif container.
#
#   sbatch truba/robo_jepa_train.slurm [config.yaml|config name]
#   sbatch truba/robo_jepa_train.slurm configs/finetuning/arena_100m_real1030c.yaml
#   sbatch --gres=gpu:4 -c 32 truba/robo_jepa_train.slurm arena_100m_real1030c   # one rank per GPU
#   DRY_RUN=1 bash truba/robo_jepa_train.slurm arena_100m_real1030c             # check paths on the login node
#
# Submit from the repository root (or set REPO_DIR). A config name is looked up as
# configs/finetuning/<name>.yaml. ${ROBO_JEPA_*} in the config are expanded from
# the variables below. Resubmitting the same config resumes from the latest
# checkpoint in its `folder`.
#
# W&B: online when WANDB_API_KEY is exported in the submitting shell or
# ~/.netrc holds an api.wandb.ai login (`wandb login` once); offline otherwise
# (sync later with `wandb sync <folder>/wandb/offline-run-*`).
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Paths (each one can be overridden from the environment)
# ------------------------------------------------------------------------------
CLUSTER_USER="${CLUSTER_USER:-yyesilnur}"
SCRATCH="${ROBO_JEPA_SCRATCH:-/arf/scratch/${CLUSTER_USER}/Workspace}"
if [[ -z "${REPO_DIR:-}" && -f "${SLURM_SUBMIT_DIR:-$PWD}/app/main.py" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
REPO_DIR="${REPO_DIR:-/arf/home/${CLUSTER_USER}/Workspace/wm/RoboJepa-Romer}"
CONTAINER="${CONTAINER:-${SCRATCH}/containers/robo_jepa.sif}"
# Host paths visible in the container (Apptainer reads APPTAINER_BIND itself).
export APPTAINER_BIND="${APPTAINER_BIND:-/arf}"

export ROBO_JEPA_WEIGHTS="${ROBO_JEPA_WEIGHTS:-${SCRATCH}/parameters/robo_jepa}"
export ROBO_JEPA_DATA="${ROBO_JEPA_DATA:-${SCRATCH}/datasets_processed/robo_jepa}"
export ROBO_JEPA_RUNS="${ROBO_JEPA_RUNS:-${SCRATCH}/parameters/robo_jepa/runs}"

# ------------------------------------------------------------------------------
# 2. Config
# ------------------------------------------------------------------------------
CONFIG_INPUT="${1:-configs/finetuning/arena_100m_real1030c.yaml}"
if [[ "${CONFIG_INPUT}" == /* && -f "${CONFIG_INPUT}" ]]; then
    CONFIG="${CONFIG_INPUT}"
elif [[ -f "${REPO_DIR}/${CONFIG_INPUT}" ]]; then
    CONFIG="${CONFIG_INPUT}"
elif [[ -f "${REPO_DIR}/configs/finetuning/${CONFIG_INPUT}.yaml" ]]; then
    CONFIG="configs/finetuning/${CONFIG_INPUT}.yaml"
else
    echo "ERROR: config ${CONFIG_INPUT} not found under ${REPO_DIR}" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. GPUs: one rank per allocated GPU. app/main.py sets CUDA_VISIBLE_DEVICES of
#    each rank from --devices, so pass the ids Slurm gave this job.
# ------------------------------------------------------------------------------
IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
DEVICES=()
for g in "${GPU_IDS[@]}"; do
    DEVICES+=("cuda:${g}")
done

# ------------------------------------------------------------------------------
# 4. W&B (no key is stored in the repository)
# ------------------------------------------------------------------------------
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -qs "api.wandb.ai" "${HOME}/.netrc"; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

echo "=================================================================="
echo "robo_jepa training on TRUBA"
echo "Job ID     : ${SLURM_JOB_ID:-N/A}"
echo "Node       : $(hostname)"
echo "Config     : ${CONFIG}"
echo "Repository : ${REPO_DIR}"
echo "Container  : ${CONTAINER}"
echo "Weights    : ${ROBO_JEPA_WEIGHTS}"
echo "Data       : ${ROBO_JEPA_DATA}"
echo "Runs       : ${ROBO_JEPA_RUNS}"
echo "Devices    : ${DEVICES[*]}"
echo "W&B mode   : ${WANDB_MODE}"
echo "Started at : $(date)"
echo "=================================================================="

for path in "$CONTAINER" "$REPO_DIR/app/main.py"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: $path does not exist" >&2
        exit 1
    fi
done

# Resolve the config inside the container (same expansion as app/main.py) and
# check that every path it reads exists before asking for the GPU time.
apptainer exec --pwd "$REPO_DIR" "$CONTAINER" python - "$CONFIG" <<'PYCHECK'
import os, sys, yaml
cfg = yaml.safe_load(os.path.expandvars(open(sys.argv[1]).read()))
data = cfg["data"]
paths = {
    "encoder": cfg["model"]["encoder_pretrained_path"],
    "checkpoint": cfg["meta"].get("read_checkpoint") or "",
}
for name, ds in data["train"].items():
    if isinstance(ds, dict) and "s3_prefix" in ds:
        paths[f"dataset {name}"] = os.path.join(data["s3_prefix"], ds["s3_prefix"], "dataset_index.csv")
print(f"folder     : {cfg['folder']}")
missing = []
for name, path in paths.items():
    ok = os.path.exists(path)
    print(f"{name:11s}: {path} {'ok' if ok else 'MISSING'}")
    if not ok and (name != "checkpoint" or cfg["meta"].get("load_checkpoint")):
        missing.append(path)
if "${" in yaml.safe_dump(cfg):
    sys.exit("ERROR: unexpanded ${...} left in the config")
if missing:
    sys.exit(f"ERROR: missing {missing}")
PYCHECK

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: paths check out, not starting training"
    exit 0
fi

# Inside a Slurm allocation the command runs through srun. app/robo_jepa/train.py pins
# CUDA_VISIBLE_DEVICES to SLURM_LOCALID when it is set, which would put every rank
# of this single task on the same GPU: unset it, app/main.py assigns the devices.
LAUNCHER=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER=(srun)
fi

set +e
"${LAUNCHER[@]}" apptainer exec --nv \
    --pwd "$REPO_DIR" \
    --env "ROBO_JEPA_WEIGHTS=${ROBO_JEPA_WEIGHTS}" \
    --env "ROBO_JEPA_DATA=${ROBO_JEPA_DATA}" \
    --env "ROBO_JEPA_RUNS=${ROBO_JEPA_RUNS}" \
    --env "WANDB_MODE=${WANDB_MODE}" \
    --env "PYTHONNOUSERSITE=1" \
    "$CONTAINER" bash -c 'unset SLURM_LOCALID; exec python -m app.main --fname "$0" --devices "$@"' \
    "$CONFIG" "${DEVICES[@]}"
RC=$?
set -e

echo "=================================================================="
echo "Training finished at: $(date)  Exit code: $RC"
echo "=================================================================="
exit $RC
