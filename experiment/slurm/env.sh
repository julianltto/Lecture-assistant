export PROJ="${PROJ:-$HOME/la_teacher_grpo}"
export LA_WORK="${LA_WORK:-$HOME}"
export HF_HOME="$LA_WORK/hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-la_teacher_grpo}"
export TEX="$PROJ/data/Linear_Algebra.tex"
export DATA="$LA_WORK/la_data"
export RUNS="$LA_WORK/la_runs"
mkdir -p "$DATA" "$RUNS" "$PROJ/logs"
cd "$PROJ"
source .venv/bin/activate

wait_http() {
  for _ in $(seq 1 180); do
    curl -sf "$1" >/dev/null && return 0
    sleep 10
  done
  echo "timeout waiting for $1" >&2
  return 1
}
