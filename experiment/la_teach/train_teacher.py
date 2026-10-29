import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from la_teach.common import load_sections, read_jsonl, render, seeded_rng, strip_think, teacher_messages
from la_teach.student_client import StudentClient


def build_rows(sections, train, tok, rows_per_section, n_examples, n_eval, max_words, seed):
    by_sec = defaultdict(list)
    for p in train:
        by_sec[p["sec"]].append(p)
    rows = []
    for s in sections:
        pool = by_sec.get(s["sec"], [])
        if len(pool) < n_examples + n_eval:
            print(f"skip sec {s['sec']}: only {len(pool)} train problems", flush=True)
            continue
        for r in range(rows_per_section):
            rng = seeded_rng(seed, "row", s["sec"], r)
            picked = rng.sample(pool, n_examples + n_eval)
            ex, ev = picked[:n_examples], picked[n_examples:]
            rows.append({
                "prompt": render(tok, teacher_messages(s, ex, max_words), enable_thinking=False),
                "sec": s["sec"],
                "eval_ids": [p["id"] for p in ev],
            })
    seeded_rng(seed, "shuffle").shuffle(rows)
    return rows


class StudentReward:
    def __init__(self, client, problems_by_id, log_path, max_fail_frac=0.2):
        self.__name__ = "student_acc"
        self.client = client
        self.problems = problems_by_id
        self.log_path = log_path
        self.max_fail_frac = max_fail_frac
        self.calls = 0

    def __call__(self, prompts, completions, eval_ids, sec, **kwargs):
        explanations = [strip_think(c if isinstance(c, str) else c[-1]["content"]) for c in completions]
        jobs, owner = [], []
        for i, (expl, ids) in enumerate(zip(explanations, eval_ids)):
            for pid in ids:
                jobs.append((self.problems[pid], expl, "Lernhilfe"))
                owner.append(i)
        self.client.reset_stats()
        accs = self.client.accuracy(jobs)
        stats = dict(self.client.stats)
        if stats["failed"] > self.max_fail_frac * max(1, stats["requests"]):
            raise RuntimeError(f"student server failing: {stats}")
        totals = [0.0] * len(explanations)
        counts = [0] * len(explanations)
        for i, a in zip(owner, accs):
            totals[i] += a
            counts[i] += 1
        rewards = [t / max(1, c) for t, c in zip(totals, counts)]
        self.calls += 1
        if self.log_path and int(os.environ.get("RANK", "0")) == 0:
            best = max(range(len(rewards)), key=rewards.__getitem__)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "call": self.calls,
                    "rewards": rewards,
                    "sec": sec[best],
                    "best_reward": rewards[best],
                    "best_explanation": explanations[best],
                    "words": [len(e.split()) for e in explanations],
                    "student": stats,
                }, ensure_ascii=False) + "\n")
        return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-14B")
    ap.add_argument("--student", default="Qwen/Qwen3-8B")
    ap.add_argument("--student_url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--student_n", type=int, default=1)
    ap.add_argument("--student_max_tokens", type=int, default=2048)
    ap.add_argument("--student_thinking", action="store_true")
    ap.add_argument("--student_max_model_len", type=int, default=12288)
    ap.add_argument("--teacher_max_model_len", type=int, default=12288)
    ap.add_argument("--vllm_host", default="127.0.0.1")
    ap.add_argument("--vllm_port", type=int, default=8000)
    ap.add_argument("--rows_per_section", type=int, default=24)
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=500)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_to", default="wandb,tensorboard")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    sections = load_sections(args.tex)
    train = read_jsonl(args.train)
    rows = build_rows(sections, train, tok, args.rows_per_section, args.n_examples, args.n_eval, args.max_words, args.seed)
    max_prompt = max(len(tok(r["prompt"], add_special_tokens=False).input_ids) for r in rows)
    print(f"{len(rows)} prompts, longest prompt {max_prompt} tokens", flush=True)
    if max_prompt + args.max_completion_length > args.teacher_max_model_len:
        raise SystemExit(f"teacher budget exceeded: prompt {max_prompt} + completion {args.max_completion_length} "
                         f"> teacher_max_model_len {args.teacher_max_model_len}")

    student_tok = AutoTokenizer.from_pretrained(args.student)
    train_by_len = sorted(train, key=lambda p: len(p["frage"]) + sum(len(o) for o in p.get("optionen", [])))
    filler = student_tok.decode([student_tok.encode(" Wort", add_special_tokens=False)[0]] * args.max_completion_length)
    probe = StudentClient(args.student_url, args.student, max_tokens=args.student_max_tokens,
                          thinking=args.student_thinking, max_context=args.student_max_model_len, tokenizer=student_tok)
    worst = max(probe.prompt_tokens(probe.fit(p, filler, "Lernhilfe")) for p in train_by_len[-20:])
    print(f"student worst case: prompt {worst} + answer {args.student_max_tokens} "
          f"<= {args.student_max_model_len}, ctx truncations {probe.stats['ctx_truncated']}", flush=True)
    if probe.stats["ctx_truncated"]:
        raise SystemExit("a full-length explanation does not fit the student context; raise student_max_model_len "
                         "or lower max_completion_length / student_max_tokens")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    client = StudentClient(args.student_url, args.student, n=args.student_n,
                           max_tokens=args.student_max_tokens, thinking=args.student_thinking,
                           max_context=args.student_max_model_len, tokenizer=student_tok)
    reward = StudentReward(client, {p["id"]: p for p in train}, str(out / "samples.jsonl"))

    config = GRPOConfig(
        output_dir=str(out),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_prompt_length=max_prompt + 16,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        logging_steps=1,
        save_steps=args.save_steps,
        save_only_model=True,
        report_to=args.report_to.split(","),
        run_name=out.name,
        log_completions=True,
        num_completions_to_print=0,
        seed=args.seed,
        model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=args.lora_targets.split(","),
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=args.teacher,
        reward_funcs=[reward],
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out / "final"))


if __name__ == "__main__":
    main()
