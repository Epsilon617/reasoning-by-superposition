#!/bin/bash
#SBATCH --job-name=fbc-cf-all-n50
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

ROOT="/mnt/iusers01/fatpou01/compsci01/r90629yl/src/LLM_Token_Info/reasoning-by-superposition"
ENV_PREFIX="/mnt/iusers01/fatpou01/compsci01/r90629yl/.conda/envs/atrain"
PYTHON_BIN="$ENV_PREFIX/bin/python"
INPUT_DIR="$ROOT/pilot/thought_trajectories_checkpoint300"
OUTPUT_DIR="$INPUT_DIR/counterfactual_patch_all_n50"
CHECKPOINT_PATH="$ROOT/ckpts_followup/prosqa-coconut-2l-8h-768d-followup-20260407/checkpoint_300"

mkdir -p "$ROOT/logs/slurm" "$OUTPUT_DIR"

export PATH="$ENV_PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export WANDB_MODE=disabled

cd "$ROOT"

"$PYTHON_BIN" -V

CMD=(
  "$PYTHON_BIN" "$ROOT/counterfactual_patch_positive_control.py"
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --checkpoint-path "$CHECKPOINT_PATH"
  --device cpu
  --patch-mode all
  --limit-pairs 50
  --n-jobs 16
)

echo "[counterfactual-patch-all-n50-cpu] Command: ${CMD[*]}"
"${CMD[@]}"
