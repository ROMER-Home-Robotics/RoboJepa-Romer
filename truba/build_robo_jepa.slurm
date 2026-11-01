#!/bin/bash
#SBATCH --job-name=build_robo_jepa
#SBATCH --output=build_robo_jepa_%j.out
#SBATCH --error=build_robo_jepa_%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH -A romer
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# The build needs no GPU, but kolyoz-cuda rejects jobs that don't request one;
# the allocated GPU is used for a quick check of the new image.
#SBATCH --gres=gpu:1

# Builds robo_jepa.sif on TRUBA from truba/robo_jepa.def (the alternative to
# building it on a workstation and copying it over):
#   cd <repo> && sbatch truba/build_robo_jepa.slurm
set -euo pipefail

CLUSTER_USER="${CLUSTER_USER:-yyesilnur}"
CONTAINER_DIR="${CONTAINER_DIR:-/arf/scratch/${CLUSTER_USER}/Workspace/containers}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DEF_FILE="${REPO_DIR}/truba/robo_jepa.def"
SIF_FILE="${CONTAINER_DIR}/robo_jepa.sif"

# Keep Apptainer's scratch off the compute node's small /tmp.
export APPTAINER_TMPDIR="${CONTAINER_DIR}/apptainer_tmp"
export APPTAINER_CACHEDIR="${CONTAINER_DIR}/apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "=== robo_jepa container build ==="
echo "node: $(hostname)"
echo "def:  ${DEF_FILE}"
echo "sif:  ${SIF_FILE}"
date

# Build to a temporary path and only swap it in once the build AND its %test
# have passed, so a failed rebuild leaves the previous image intact.
TMP_SIF="${SIF_FILE}.building.$$"
rm -f "${TMP_SIF}"
apptainer build --fakeroot "${TMP_SIF}" "${DEF_FILE}"

echo "=== running %test against the new image ==="
apptainer test "${TMP_SIF}"

echo "=== GPU check ==="
apptainer exec --nv "${TMP_SIF}" python -c "
import torch
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device='cuda'); print('matmul ok', float((x @ x).sum()))
"

mv -f "${TMP_SIF}" "${SIF_FILE}"
echo "=== installed ${SIF_FILE} ==="
date
