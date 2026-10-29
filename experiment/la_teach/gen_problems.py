import argparse
import re
from collections import defaultdict
from pathlib import Path

from la_teach.common import (
    answers_match,
    last_boxed,
    load_sections,
    render,
    seeded_rng,
    shuffle_mc,
    strip_think,
    student_messages,
    write_jsonl,
)

GEN_INSTRUCTIONS = """Erstelle {k} neue, abwechslungsreiche Übungsaufgaben zu dem obigen Abschnitt {sec} ({title}).
Schwerpunkt dieser Serie: {focus}

Anforderungen:
- Jede Aufgabe hat eine eindeutige, kurze, automatisch überprüfbare Endantwort.
- Erlaubte Typen:
  * wert: Endantwort ist ein konkretes mathematisches Objekt (Zahl, Vektor, Matrix, Polynom, endliche Menge), als LaTeX ohne Dollarzeichen.
  * mc: genau 4 Antwortoptionen, genau eine ist richtig; Antwort ist der Buchstabe A, B, C oder D.
  * wahrfalsch: eine Aussage; Antwort ist wahr oder falsch.
- Bevorzuge den Typ wert, wann immer der Stoff konkrete Rechnungen oder Beispiele zulässt. Höchstens eine Aufgabe vom Typ wahrfalsch.
- Keine Beweisaufgaben. Die Aufgaben sollen Verständnis und Anwenden des Stoffs prüfen, nicht wörtliches Zitieren des Skripts.
- Mische die Schwierigkeit: einige einfache, mehrere mittlere und einige anspruchsvolle Aufgaben mit mehreren Schritten.
- Die Aufgaben müssen ohne das Skript verständlich formuliert sein.

Ausgabeformat, für jede Aufgabe genau so:
<aufgabe>
<typ>wert|mc|wahrfalsch</typ>
<frage>...</frage>
<option>...</option>   (nur bei mc, genau 4 Zeilen in der Reihenfolge A, B, C, D)
<antwort>...</antwort>
<loesung>kurzer Lösungsweg</loesung>
</aufgabe>
Gib nichts anderes aus."""

FOCI = [
    "konkrete Rechenbeispiele",
    "Anwendung der Sätze auf neue Beispiele",
    "Gegenbeispiele und typische Fehlvorstellungen",
    "Verständnis der Definitionen an konkreten Objekten",
    "mehrschrittige Aufgaben, die mehrere Resultate des Abschnitts kombinieren",
    "Sonderfälle und Grenzfälle",
]

AUFGABE_RE = re.compile(r"<aufgabe>(.*?)</aufgabe>", re.S)


def tag(block, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", block, re.S)
    return m.group(1).strip() if m else None


def parse_problems(text):
    out = []
    for block in AUFGABE_RE.findall(text):
        typ = (tag(block, "typ") or "").lower().strip()
        frage = tag(block, "frage")
        antwort = tag(block, "antwort")
        loesung = tag(block, "loesung") or ""
        if not frage or not antwort or typ not in {"wert", "mc", "wahrfalsch"}:
            continue
        p = {"typ": typ, "frage": frage, "antwort": antwort.replace("$", "").strip(), "loesung": loesung}
        if typ == "mc":
            opts = [o.strip() for o in re.findall(r"<option>(.*?)</option>", block, re.S)]
            letter = re.sub(r"[^A-D]", "", p["antwort"].upper())[:1]
            if len(opts) != 4 or not letter or len(set(opts)) != 4:
                continue
            p["optionen"] = opts
            p["antwort"] = letter
        if typ == "wahrfalsch":
            a = p["antwort"].lower()
            if a.startswith("wahr"):
                p["antwort"] = "wahr"
            elif a.startswith("falsch"):
                p["antwort"] = "falsch"
            else:
                continue
        out.append(p)
    return out


def ngrams(s, n=5):
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def dedupe(problems, thr):
    kept, grams = [], []
    for p in problems:
        g = ngrams(p["frage"] + " ".join(p.get("optionen", [])))
        if any(len(g & h) / max(1, len(g | h)) > thr for h in grams):
            continue
        kept.append(p)
        grams.append(g)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--prompts_per_section", type=int, default=48)
    ap.add_argument("--per_prompt", type=int, default=8)
    ap.add_argument("--n_verify", type=int, default=3)
    ap.add_argument("--min_verify", type=int, default=2)
    ap.add_argument("--dedupe_thr", type=float, default=0.6)
    ap.add_argument("--sections", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    sections = load_sections(args.tex)
    if args.sections:
        keep = {int(x) for x in args.sections.split(",")}
        sections = [s for s in sections if s["sec"] in keep]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    tok = llm.get_tokenizer()

    prompts, meta = [], []
    for s in sections:
        items = s["items"] or ["den gesamten Abschnitt"]
        for j in range(args.prompts_per_section):
            focus = f"{items[j % len(items)]}; Aufgabenstil: {FOCI[j % len(FOCI)]}"
            msg = [{"role": "user", "content": f"Skriptabschnitt:\n\n{s['text']}\n\n---\n\n" + GEN_INSTRUCTIONS.format(
                k=args.per_prompt, sec=s["sec"], title=s["title"], focus=focus)}]
            prompts.append(render(tok, msg))
            meta.append(s["sec"])
    gen = llm.generate(prompts, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=6000, seed=args.seed))

    by_sec = defaultdict(list)
    for sec, o in zip(meta, gen):
        for p in parse_problems(strip_think(o.outputs[0].text)):
            by_sec[sec].append(p)

    candidates = []
    for s in sections:
        rng = seeded_rng(args.seed, s["sec"])
        ps = dedupe(by_sec[s["sec"]], args.dedupe_thr)
        for i, p in enumerate(ps):
            p = shuffle_mc(p, rng)
            p["sec"] = s["sec"]
            p["id"] = f"s{s['sec']:02d}_{i:04d}"
            candidates.append(p)
        print(f"sec {s['sec']:2d}: raw {len(by_sec[s['sec']])}, dedup {len(ps)}", flush=True)

    sec_text = {s["sec"]: s["text"] for s in sections}
    vprompts = [render(tok, student_messages(p, sec_text[p["sec"]], "Skriptauszug")) for p in candidates]
    vout = llm.generate(vprompts, SamplingParams(n=args.n_verify, temperature=0.7, top_p=0.95, max_tokens=3000, seed=args.seed + 1))

    verified = []
    for p, o in zip(candidates, vout):
        hits = sum(answers_match(p["antwort"], last_boxed(strip_think(c.text)), p["typ"]) for c in o.outputs)
        p["verify_rate"] = hits / args.n_verify
        if hits >= args.min_verify:
            verified.append(p)

    out = Path(args.out)
    write_jsonl(out.with_name(out.stem + "_all.jsonl"), candidates)
    write_jsonl(out, verified)
    counts = defaultdict(int)
    for p in verified:
        counts[p["sec"]] += 1
    for s in sections:
        print(f"sec {s['sec']:2d} {s['title'][:40]:40s} verified {counts[s['sec']]}", flush=True)


if __name__ == "__main__":
    main()
