import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from la_teach.common import (
    grade,
    load_sections,
    read_jsonl,
    render,
    seeded_rng,
    strip_think,
    student_messages,
    teacher_messages,
    write_jsonl,
)


def cmd_write(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sections = load_sections(args.tex)
    train = read_jsonl(args.train)
    by_sec = defaultdict(list)
    for p in train:
        by_sec[p["sec"]].append(p)

    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, max_model_len=12288,
              gpu_memory_utilization=0.9, enable_lora=bool(args.adapter), max_lora_rank=args.max_lora_rank,
              seed=args.seed)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for s in sections:
        pool = by_sec.get(s["sec"], [])
        if len(pool) < args.n_examples:
            continue
        for k in range(args.n_expl):
            ex = seeded_rng(args.seed, "evalex", s["sec"], k).sample(pool, args.n_examples)
            prompts.append(render(tok, teacher_messages(s, ex, args.max_words), enable_thinking=False))
            meta.append((s["sec"], k))
    lora = LoRARequest("teacher", 1, args.adapter) if args.adapter else None
    outs = llm.generate(prompts, SamplingParams(temperature=args.temperature, top_p=1.0,
                                                max_tokens=args.max_tokens, seed=args.seed), lora_request=lora)
    rows = [{"sec": sec, "k": k, "explanation": strip_think(o.outputs[0].text)} for (sec, k), o in zip(meta, outs)]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} explanations to {args.out}")


def bootstrap_ci(diffs, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def cmd_solve(args):
    from vllm import LLM, SamplingParams

    sections = {s["sec"]: s for s in load_sections(args.tex)}
    test = read_jsonl(args.test)
    conditions = {"none": None, "notes": "notes"}
    for spec in args.expl:
        tag, path = spec.split("=", 1)
        by_sec = defaultdict(list)
        for r in read_jsonl(path):
            by_sec[r["sec"]].append(r["explanation"])
        conditions[tag] = by_sec

    llm = LLM(model=args.student, tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.9, enable_prefix_caching=True, seed=args.seed)
    tok = llm.get_tokenizer()

    prompts, owner = [], []
    for cname, cond in conditions.items():
        for qi, p in enumerate(test):
            if cond is None:
                ctxs = [(None, None)] * args.n_none
            elif cond == "notes":
                ctxs = [(sections[p["sec"]]["text"], "Skriptauszug")] * args.n_none
            else:
                ctxs = [(e, "Lernhilfe") for e in cond.get(p["sec"], [])]
            for ctx, label in ctxs:
                prompts.append(render(tok, student_messages(p, ctx, label), args.student_thinking))
                owner.append((cname, qi))
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.8, max_tokens=args.max_tokens))

    hits = defaultdict(list)
    for (cname, qi), o in zip(owner, outs):
        hits[(cname, qi)].append(grade(test[qi], strip_think(o.outputs[0].text)))
    acc = {c: [sum(hits[(c, qi)]) / len(hits[(c, qi)]) if hits[(c, qi)] else float("nan") for qi in range(len(test))]
           for c in conditions}

    result = {"n_test": len(test), "overall": {}, "vs_none": {}, "per_section": {}}
    for c in conditions:
        vals = [a for a in acc[c] if a == a]
        result["overall"][c] = round(sum(vals) / max(1, len(vals)), 4)
        if c != "none":
            d = [a - b for a, b in zip(acc[c], acc["none"]) if a == a]
            lo, hi = bootstrap_ci(d) if d else (float("nan"), float("nan"))
            result["vs_none"][c] = {"diff": round(sum(d) / max(1, len(d)), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for sec in sorted({p["sec"] for p in test}):
        idx = [i for i, p in enumerate(test) if p["sec"] == sec]
        result["per_section"][sec] = {c: round(sum(acc[c][i] for i in idx) / len(idx), 3) for c in conditions}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "vs_none": result["vs_none"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write")
    w.add_argument("--tex", required=True)
    w.add_argument("--train", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--teacher", default="Qwen/Qwen3-14B")
    w.add_argument("--adapter", default="")
    w.add_argument("--max_lora_rank", type=int, default=64)
    w.add_argument("--n_expl", type=int, default=4)
    w.add_argument("--n_examples", type=int, default=2)
    w.add_argument("--max_words", type=int, default=500)
    w.add_argument("--max_tokens", type=int, default=1024)
    w.add_argument("--temperature", type=float, default=0.7)
    w.add_argument("--tp", type=int, default=1)
    w.add_argument("--seed", type=int, default=1234)

    s = sub.add_parser("solve")
    s.add_argument("--tex", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--expl", nargs="*", default=[], help="tag=path.jsonl")
    s.add_argument("--student", default="Qwen/Qwen3-8B")
    s.add_argument("--student_thinking", action="store_true")
    s.add_argument("--n_none", type=int, default=4)
    s.add_argument("--max_tokens", type=int, default=2048)
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()
    {"write": cmd_write, "solve": cmd_solve}[args.cmd](args)


if __name__ == "__main__":
    main()
