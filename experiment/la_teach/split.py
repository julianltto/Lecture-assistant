import argparse
import json
from collections import defaultdict
from pathlib import Path

from la_teach.common import grade, read_jsonl, render, seeded_rng, strip_think, student_messages, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--student", default="Qwen/Qwen3-8B")
    ap.add_argument("--student_thinking", action="store_true")
    ap.add_argument("--n_solo", type=int, default=4)
    ap.add_argument("--max_solo", type=float, default=0.75)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    problems = read_jsonl(args.problems)
    llm = LLM(model=args.student, tensor_parallel_size=args.tp, max_model_len=8192,
              gpu_memory_utilization=0.9, enable_prefix_caching=True, seed=args.seed)
    tok = llm.get_tokenizer()
    prompts = [render(tok, student_messages(p), args.student_thinking) for p in problems]
    outs = llm.generate(prompts, SamplingParams(n=args.n_solo, temperature=0.7, top_p=0.8,
                                                max_tokens=args.max_tokens, seed=args.seed))
    for p, o in zip(problems, outs):
        p["solo_rate"] = sum(grade(p, strip_think(c.text)) for c in o.outputs) / args.n_solo

    by_sec = defaultdict(list)
    for p in problems:
        by_sec[p["sec"]].append(p)

    train, test, report = [], [], {}
    for sec in sorted(by_sec):
        pool = [p for p in by_sec[sec] if p["solo_rate"] <= args.max_solo]
        seeded_rng(args.seed, "split", sec).shuffle(pool)
        need = args.n_train + args.n_test
        if len(pool) >= need:
            tr, te = pool[:args.n_train], pool[args.n_train:need]
        else:
            half = len(pool) // 2
            tr, te = pool[:half], pool[half:2 * half]
        train += tr
        test += te
        report[sec] = {
            "verified": len(by_sec[sec]),
            "after_filter": len(pool),
            "train": len(tr),
            "test": len(te),
            "solo_all": round(sum(p["solo_rate"] for p in by_sec[sec]) / max(1, len(by_sec[sec])), 3),
            "solo_test": round(sum(p["solo_rate"] for p in te) / max(1, len(te)), 3),
            "types_test": {t: sum(p["typ"] == t for p in te) for t in ("wert", "mc", "wahrfalsch")},
        }
        print(sec, report[sec], flush=True)

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "problems_with_solo.jsonl", problems)
    (out / "split_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
