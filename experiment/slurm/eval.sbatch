#!/bin/bash
#SBATCH --job-name=la_eval
#SBATCH --account=ag_ifi_wrobel
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

source "${PROJ:-$HOME/la_teacher_grpo}/slurm/env.sh"

TEACHER="${TEACHER:-Qwen/Qwen3-14B}"
STUDENT="${STUDENT:-Qwen/Qwen3-8B}"
OUT="${OUT:-$RUNS/eval_$(date +%m%d_%H%M)}"
mkdir -p "$OUT"

python -m la_teach.eval_teacher write --tex "$TEX" --train "$DATA/train.jsonl" \
  --teacher "$TEACHER" --out "$OUT/expl_base.jsonl" || exit 1
EXPL=("base=$OUT/expl_base.jsonl")

if [ -n "$ADAPTER" ]; then
  python -m la_teach.eval_teacher write --tex "$TEX" --train "$DATA/train.jsonl" \
    --teacher "$TEACHER" --adapter "$ADAPTER" --out "$OUT/expl_trained.jsonl" || exit 1
  EXPL+=("trained=$OUT/expl_trained.jsonl")
fi

python -m la_teach.eval_teacher solve --tex "$TEX" --test "$DATA/test.jsonl" \
  --student "$STUDENT" --expl "${EXPL[@]}" --out "$OUT/results.json"
