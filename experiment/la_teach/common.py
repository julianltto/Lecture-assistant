import json
import random
import re
from pathlib import Path

SECTION_RE = re.compile(r"\\section\*?\{((?:[^{}]|\{[^{}]*\})*)\}")
NR_RE = re.compile(r"\\nr\{([^}]*)\}\{([^}]*)\}")


def load_sections(tex_path, drop_titles=("Literatur",)):
    text = Path(tex_path).read_text(encoding="utf-8")
    text = text.split("\\end{document}")[0]
    matches = list(SECTION_RE.finditer(text))
    sections = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        body = text[m.end():end]
        body = re.sub(r"\\part\*?\{[^}]*\}", "", body)
        body = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", body)
        body = body.strip()
        title = m.group(1).strip()
        if title in drop_titles:
            continue
        items = [f"({a}) {b}" for a, b in NR_RE.findall(body)]
        sections.append({"sec": k + 1, "title": title, "text": body, "items": items})
    return sections


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


LETTERS = "ABCDEFGH"


def format_problem(p):
    if p["typ"] == "mc":
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(p["optionen"]))
        return f"{p['frage']}\n\n{opts}\n\nGib nur den Buchstaben der richtigen Option in \\boxed{{}} an."
    if p["typ"] == "wahrfalsch":
        return f"Wahr oder falsch?\n{p['frage']}\n\nGib \\boxed{{wahr}} oder \\boxed{{falsch}} an."
    return f"{p['frage']}\n\nGib die Endantwort in \\boxed{{}} an."


def teacher_messages(section, examples, max_words):
    ex = "\n\n".join(
        f"Beispielaufgabe {i + 1}:\n{format_problem(p)}\nLösung: {p['loesung']}\nAntwort: {p['antwort']}"
        for i, p in enumerate(examples)
    )
    user = (
        f"Hier ist Abschnitt {section['sec']} ({section['title']}) aus dem Skript \"Lineare Algebra\":\n\n"
        f"{section['text']}\n\n---\n\n{ex}\n\n---\n\n"
        "Schreibe eine Lernhilfe zu diesem Abschnitt für Studierende. Die Studierenden sehen den Skripttext "
        "und die Beispielaufgaben NICHT, sondern nur deine Lernhilfe. Danach müssen sie neue, ähnliche Aufgaben "
        "zu diesem Abschnitt selbstständig lösen. Erkläre die Begriffe, Sätze und Rechenverfahren, die man dafür "
        f"braucht, und typische Fehler. Höchstens {max_words} Wörter. Schreibe nur die Lernhilfe."
    )
    return [{"role": "user", "content": user}]


def student_messages(problem, context=None, context_label="Lernhilfe"):
    task = format_problem(problem)
    if context:
        content = f"{context_label}:\n{context}\n\n---\n\nAufgabe:\n{task}"
    else:
        content = f"Aufgabe:\n{task}"
    return [
        {"role": "system", "content": "Du löst Aufgaben zur linearen Algebra. Rechne kurz und sorgfältig und schreibe die Endantwort in \\boxed{}."},
        {"role": "user", "content": content},
    ]


def last_boxed(text):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


TRUE_WORDS = {"wahr", "ja", "true", "richtig", "w"}
FALSE_WORDS = {"falsch", "nein", "false", "f"}


def _strip_text(s):
    s = re.sub(r"\\text(bf|rm|it)?\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    return s.replace("$", "").strip()


def _norm(s):
    s = _strip_text(s)
    s = re.sub(r"\\[,;:! ]|\\left|\\right|\\displaystyle", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"\s+", "", s).rstrip(".").lower()


def answers_match(gold, pred, typ):
    if pred is None:
        return False
    if typ == "mc":
        m = re.search(r"[A-H]", _strip_text(pred).upper())
        return bool(m) and m.group(0) == gold.strip().upper()
    if typ == "wahrfalsch":
        p = _norm(pred).strip("()")
        g = _norm(gold)
        if p in TRUE_WORDS:
            return g in TRUE_WORDS
        if p in FALSE_WORDS:
            return g in FALSE_WORDS
        return False
    if _norm(gold) == _norm(pred):
        return True
    try:
        from math_verify import parse, verify
        g = parse(f"${gold}$")
        q = parse(f"${pred}$")
        return bool(g) and bool(q) and bool(verify(g, q))
    except Exception:
        return False


def grade(problem, response):
    return answers_match(problem["antwort"], last_boxed(response), problem["typ"])


def shuffle_mc(p, rng):
    if p["typ"] != "mc":
        return p
    opts = list(p["optionen"])
    correct = opts[LETTERS.index(p["antwort"].strip().upper())]
    rng.shuffle(opts)
    q = dict(p)
    q["optionen"] = opts
    q["antwort"] = LETTERS[opts.index(correct)]
    return q


def seeded_rng(*keys):
    return random.Random("|".join(map(str, keys)))


def render(tokenizer, messages, enable_thinking=False):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
