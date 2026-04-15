#!/bin/bash
#SBATCH --job-name=fbc-mod10-2g
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

ROOT="/mnt/iusers01/fatpou01/compsci01/r90629yl/src/LLM_Token_Info/reasoning-by-superposition"
ENV_PREFIX="/mnt/iusers01/fatpou01/compsci01/r90629yl/.conda/envs/atrain"
TRAIN_CONFIG="${1:?Usage: sbatch slurm/modadd_10way_train_2gpu.sbatch args/config.yaml}"
HF_CACHE_DIR="$ROOT/pilot/hf_cache"
MPLCONFIGDIR="${TMPDIR:-/tmp}/fbc-mod10-mplconfig-${SLURM_JOB_ID}"

mkdir -p "$ROOT/logs/slurm" "$HF_CACHE_DIR" "$MPLCONFIGDIR"

export PATH="$ENV_PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=online
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_DISABLE_TELEMETRY=1
export MPLCONFIGDIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"

nvidia-smi
python -V
wandb login --verify

echo "[modadd-10way-2gpu] Training command: torchrun --standalone --nnodes 1 --nproc_per_node 2 run.py $TRAIN_CONFIG"
torchrun --standalone --nnodes 1 --nproc_per_node 2 run.py "$TRAIN_CONFIG"
