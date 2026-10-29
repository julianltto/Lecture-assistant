#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import trl, vllm, peft, math_verify; print('trl', trl.__version__, 'vllm', vllm.__version__)"

export HF_HOME="${LA_WORK:-$HOME}/hf"
for m in ${MODELS:-Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B-Instruct-2507}; do
  hf download "$m" --exclude "*.pth" "original/*"
done
