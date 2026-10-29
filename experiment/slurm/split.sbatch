#!/bin/bash
#SBATCH --job-name=la_split
#SBATCH --account=ag_ifi_wrobel
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

source "${PROJ:-$HOME/la_teacher_grpo}/slurm/env.sh"

python -m la_teach.split \
  --problems "$DATA/problems_verified.jsonl" \
  --out_dir "$DATA" \
  --student "${STUDENT:-Qwen/Qwen3-8B}" \
  ${SPLIT_ARGS}
