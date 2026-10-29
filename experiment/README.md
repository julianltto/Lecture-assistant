# Teacher-only GRPO on Blomer's "Lineare Algebra"

Teacher (default Qwen3-14B, LoRA) sees one section of the notes plus 2 train problems and writes a "Lernhilfe".
Reward = frozen student's (Qwen3-8B, non-thinking) accuracy on 8 other train problems of the same section, seeing only the Lernhilfe.
The teacher never sees the reward problems; test problems are never used in training.

## Pipeline

| step | script | GPUs | output |
|---|---|---|---|
| 1 generate + verify problems | `slurm/gen.sbatch` (Qwen3-30B-A3B-Instruct-2507, 3x self-verification, dedupe) | 2 | `$DATA/problems_verified.jsonl` |
| 2 student solo rate, drop easy (solo > 0.75), 100/100 split per section | `slurm/split.sbatch` | 1 | `$DATA/train.jsonl`, `test.jsonl`, `split_report.json` |
| 3 baseline eval (none / raw notes / untrained teacher) | `slurm/eval.sbatch` | 1 | `$RUNS/eval_*/results.json` |
| 4 GRPO | `slurm/train.sbatch` (GPU0-1 student vLLM, GPU2-3 teacher `trl vllm-serve`, GPU4-7 training) | 8 | `$RUNS/teacher_*/` |
| 5 eval trained adapter | `ADAPTER=... sbatch slurm/eval.sbatch` | 1 | `results.json` |

Knobs are env vars: `TEACHER`, `STUDENT`, `GEN_MODEL`, `SECTIONS`, `GEN_ARGS`, `SPLIT_ARGS`, `TRAIN_ARGS`, `ADAPTER`, `LA_WORK`.

## Files to watch during training

- `$RUN/samples.jsonl`: best explanation per reward call with its reward
- `tensorboard --logdir $RUN`: `reward`, `reward_std`, `frac_reward_zero_std`, `completions/mean_length`
- `$RUN/student_server.log`, `$RUN/teacher_server.log`

## Local tests (Linux / WSL)

```
pip install math-verify pytest && python -m pytest -q tests
```
