#!/bin/bash
#SBATCH --job-name=fbc-modadd-attn
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

ROOT="/mnt/iusers01/fatpou01/compsci01/r90629yl/src/LLM_Token_Info/reasoning-by-superposition"
ENV_PREFIX="/mnt/iusers01/fatpou01/compsci01/r90629yl/.conda/envs/atrain"
PYTHON_BIN="$ENV_PREFIX/bin/python"

STANDARD_CKPT="$ROOT/ckpts_modadd/modadd-len5-mod10-fixed5-coconut-2l-8h-768d-20260413/checkpoint_120"
ISOLATED_CKPT="$ROOT/ckpts_modadd/modadd-len5-mod10-fixed5-latentonly-coconut-2l-8h-768d-20260414/checkpoint_120"
OUTPUT_DIR="$ROOT/pilot/modadd_attention_pass2_standard_vs_latentonly"

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
  "$PYTHON_BIN" "$ROOT/analyze_modadd_attention.py"
  --standard-checkpoint "$STANDARD_CKPT"
  --isolated-checkpoint "$ISOLATED_CKPT"
  --config-path "$ROOT/configs/symbol-2layer-8head-768dim.json"
  --test-path "$ROOT/data/modadd_len5_mod10_fast_test.json"
  --output-dir "$OUTPUT_DIR"
  --device cpu
  --pass-idx 2
  --limit 64
)

echo "[modadd-attention-cpu] Command: ${CMD[*]}"
"${CMD[@]}"
