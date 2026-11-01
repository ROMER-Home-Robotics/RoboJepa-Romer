#!/bin/bash
#SBATCH --output=eval-slurm-%x-%j.out
#SBATCH --error=eval-slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=kolyoz9,kolyoz13,kolyoz19,kolyoz24,kolyoz30,kolyoz31,kolyoz60
#SBATCH -p kolyoz-cuda
#SBATCH -A romer
#SBATCH --qos=midprio
#SBATCH -J robo_jepa_eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -C "H100|H200"

# ==============================================================================
# Held-out forward-prediction eval (scripts/evals/eval_arena_holdout.py) on TRUBA.
# Every argument is passed to the eval; --config and --encoder default to the
# 100M weights under ROBO_JEPA_WEIGHTS (pass them to override).
#
#   D=/arf/scratch/yyesilnur/Workspace/datasets_processed/robo_jepa
#   W=/arf/scratch/yyesilnur/Workspace/parameters/robo_jepa
#   R=$W/runs/arena_100m_real1030c
#   sbatch truba/robo_jepa_eval.slurm \
#       --data sim_heldout=$D/real1030_heldout:30 benchmark=$D/rs1030_benchmark:5 \
#       --model pretrained=$W/wm-100m/cooldown_100m_150k.pth.tar e30=$R/world_model-e30.pth.tar \
#       --out-dir $W/evals/arena_100m_real1030c
#
# Results already in --out-dir are skipped, so resubmitting with more --model
# entries only evaluates the new ones; summary.md is rewritten every time.
# ==============================================================================
set -euo pipefail

CLUSTER_USER="${CLUSTER_USER:-yyesilnur}"
SCRATCH="${ROBO_JEPA_SCRATCH:-/arf/scratch/${CLUSTER_USER}/Workspace}"
if [[ -z "${REPO_DIR:-}" && -f "${SLURM_SUBMIT_DIR:-$PWD}/app/main.py" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
REPO_DIR="${REPO_DIR:-/arf/home/${CLUSTER_USER}/Workspace/wm/RoboJepa-Romer}"
CONTAINER="${CONTAINER:-${SCRATCH}/containers/robo_jepa.sif}"
ROBO_JEPA_WEIGHTS="${ROBO_JEPA_WEIGHTS:-${SCRATCH}/parameters/robo_jepa}"
export APPTAINER_BIND="${APPTAINER_BIND:-/arf}"

echo "robo_jepa eval | job ${SLURM_JOB_ID:-N/A} | node $(hostname) | $(date)"
echo "args: $*"

LAUNCHER=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER=(srun)
fi
"${LAUNCHER[@]}" apptainer exec --nv --pwd "$REPO_DIR" --env "PYTHONPATH=${REPO_DIR}" "$CONTAINER" \
    python scripts/evals/eval_arena_holdout.py \
    --config "${ROBO_JEPA_WEIGHTS}/wm-100m/config.json" \
    --encoder "${ROBO_JEPA_WEIGHTS}/vjepa2_1_vitG_384.pt" \
    "$@"
