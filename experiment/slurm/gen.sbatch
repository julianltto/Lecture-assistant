#!/bin/bash
#SBATCH --job-name=la_gen
#SBATCH --account=ag_ifi_wrobel
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

source "${PROJ:-$HOME/la_teacher_grpo}/slurm/env.sh"

GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

python -m la_teach.gen_problems \
  --tex "$TEX" \
  --out "$DATA/problems_verified.jsonl" \
  --model "$GEN_MODEL" \
  --tp 2 \
  ${SECTIONS:+--sections "$SECTIONS"} \
  ${GEN_ARGS}
