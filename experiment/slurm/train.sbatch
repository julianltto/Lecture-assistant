#!/bin/bash
#SBATCH --job-name=la_train
#SBATCH --account=ag_ifi_wrobel
#SBATCH --partition=mlgpu_medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

source "${PROJ:-$HOME/la_teacher_grpo}/slurm/env.sh"

TEACHER="${TEACHER:-Qwen/Qwen3-14B}"
STUDENT="${STUDENT:-Qwen/Qwen3-8B}"
RUN="${RUN:-$RUNS/teacher_$(date +%m%d_%H%M)}"
mkdir -p "$RUN"
export WANDB_DIR="$RUN"

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$STUDENT" \
  --port 8001 --data-parallel-size 2 --max-model-len 12288 \
  --enable-prefix-caching --gpu-memory-utilization 0.9 \
  > "$RUN/student_server.log" 2>&1 &
STUDENT_PID=$!

CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve --model "$TEACHER" \
  --port 8000 --tensor_parallel_size 2 --max_model_len 12288 \
  --gpu_memory_utilization 0.9 --enable_prefix_caching true \
  > "$RUN/teacher_server.log" 2>&1 &
TEACHER_PID=$!

trap 'kill $STUDENT_PID $TEACHER_PID 2>/dev/null' EXIT

wait_http http://127.0.0.1:8001/health || exit 1
wait_http http://127.0.0.1:8000/health/ || exit 1

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file configs/zero3.yaml --num_processes 4 \
  -m la_teach.train_teacher \
  --tex "$TEX" \
  --train "$DATA/train.jsonl" \
  --output_dir "$RUN" \
  --teacher "$TEACHER" \
  --student "$STUDENT" \
  ${TRAIN_ARGS}
