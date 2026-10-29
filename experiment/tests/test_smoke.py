import json
import random
from pathlib import Path

from la_teach.common import (
    format_problem,
    grade,
    last_boxed,
    load_sections,
    shuffle_mc,
    student_messages,
    teacher_messages,
)
from la_teach.gen_problems import dedupe, parse_problems
from la_teach.train_teacher import StudentReward, build_rows

TEX = Path(__file__).resolve().parents[1] / "data" / "Linear_Algebra.tex"

GEN_OUTPUT = r"""
<aufgabe>
<typ>wert</typ>
<frage>Berechne die Determinante von $\begin{pmatrix}1&2\\3&4\end{pmatrix}$.</frage>
<antwort>$-2$</antwort>
<loesung>1*4-2*3=-2</loesung>
</aufgabe>
<aufgabe>
<typ>mc</typ>
<frage>Welche Dimension hat $\mathbb{R}^{2\times 3}$?</frage>
<option>5</option>
<option>6</option>
<option>3</option>
<option>9</option>
<antwort>B</antwort>
<loesung>2*3</loesung>
</aufgabe>
<aufgabe>
<typ>wahrfalsch</typ>
<frage>Jede invertierbare Matrix ist quadratisch.</frage>
<antwort>Wahr</antwort>
<loesung>Definition</loesung>
</aufgabe>
<aufgabe>
<typ>mc</typ>
<frage>kaputt</frage>
<option>1</option>
<antwort>A</antwort>
</aufgabe>
"""


def test_sections():
    secs = load_sections(TEX)
    assert len(secs) == 34
    assert secs[0]["title"] == "Mengen" and secs[-1]["sec"] == 34
    assert secs[12]["title"] == "Matrizen" and len(secs[12]["items"]) == 19
    assert "\\section" not in secs[0]["text"]
    assert all(s["text"] for s in secs)


def test_parse_and_grade():
    ps = parse_problems(GEN_OUTPUT)
    assert [p["typ"] for p in ps] == ["wert", "mc", "wahrfalsch"]
    assert ps[0]["antwort"] == "-2" and ps[2]["antwort"] == "wahr"
    mc = shuffle_mc(ps[1], random.Random(3))
    assert mc["optionen"][ord(mc["antwort"]) - 65] == "6"
    assert grade(ps[0], "also \\boxed{-2}")
    assert grade(ps[0], "\\boxed{ -2. }")
    assert not grade(ps[0], "\\boxed{2}")
    assert grade(mc, f"Antwort: \\boxed{{{mc['antwort']}}}")
    assert grade(mc, f"\\boxed{{\\text{{({mc['antwort']})}}}}")
    assert not grade(mc, "\\boxed{Z}") and not grade(mc, "keine Antwort")
    assert grade(ps[2], "\\boxed{wahr}") and grade(ps[2], "\\boxed{\\text{Ja}}")
    assert not grade(ps[2], "\\boxed{falsch}")
    assert "\\boxed{}" in format_problem(mc)


def test_math_verify_equivalence():
    p = {"typ": "wert", "antwort": "\\frac{1}{2}"}
    assert grade(p, "\\boxed{0.5}")
    assert grade(p, "\\boxed{\\dfrac{1}{2}}")
    m = {"typ": "wert", "antwort": "\\begin{pmatrix}1&0\\\\0&1\\end{pmatrix}"}
    assert grade(m, "\\boxed{\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}}")
    assert not grade(m, "\\boxed{\\begin{pmatrix}1&1\\\\0&1\\end{pmatrix}}")
    s = {"typ": "wert", "antwort": "\\{1,2,3\\}"}
    assert grade(s, "\\boxed{\\{3,2,1\\}}")
    assert last_boxed("a \\boxed{x^{2}} b \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_dedupe():
    a = {"frage": "Berechne die Determinante der Matrix A = [[1,2],[3,4]]."}
    b = {"frage": "Berechne die Determinante der Matrix A = [[1,2],[3,5]]."}
    c = {"frage": "Ist die Abbildung f(x)=x^2 injektiv auf R?"}
    assert len(dedupe([a, b, c], 0.6)) == 2


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<|user|>" + messages[-1]["content"] + "<|assistant|>"


class FakeClient:
    def __init__(self):
        self.jobs = []

    def accuracy(self, jobs):
        self.jobs += jobs
        return [1.0 if "GUT" in ctx else 0.0 for _, ctx, _ in jobs]


def make_train(secs, per=20):
    train = []
    for s in secs:
        for i in range(per):
            train.append({"id": f"s{s['sec']:02d}_{i:04d}", "sec": s["sec"], "typ": "wert",
                          "frage": f"Frage {i}", "antwort": str(i), "loesung": "x"})
    return train


def test_rows_and_reward(tmp_path):
    secs = load_sections(TEX)[:3]
    train = make_train(secs)
    train = [p for p in train if not (p["sec"] == 3 and int(p["id"][-4:]) >= 5)]
    rows = build_rows(secs, train, FakeTok(), rows_per_section=4, n_examples=2, n_eval=8, max_words=500, seed=0)
    assert len(rows) == 8 and {r["sec"] for r in rows} == {1, 2}
    by_id = {p["id"]: p for p in train}
    for r in rows:
        assert len(r["eval_ids"]) == 8 and all(by_id[i]["sec"] == r["sec"] for i in r["eval_ids"])
        ex_qs = [p["frage"] for p in train if p["sec"] == r["sec"] and f"Lösung: x\nAntwort: {p['antwort']}\n" in r["prompt"] + "\n"]
        assert len(ex_qs) == 2
        assert not any(f"Beispielaufgabe" in by_id[i]["frage"] for i in r["eval_ids"])
    client = FakeClient()
    reward = StudentReward(client, by_id, str(tmp_path / "s.jsonl"))
    out = reward(prompts=["p", "p"], completions=["<think>x</think>GUT erklärt", "schlecht"],
                 eval_ids=[rows[0]["eval_ids"]] * 2, sec=[rows[0]["sec"]] * 2)
    assert out == [1.0, 0.0]
    assert len(client.jobs) == 16 and client.jobs[0][1] == "GUT erklärt"
    log = json.loads((tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert log["best_reward"] == 1.0
    out2 = reward(prompts=["p"], completions=[[{"role": "assistant", "content": "GUT"}]],
                  eval_ids=[rows[0]["eval_ids"]], sec=[1])
    assert out2 == [1.0]


def test_prompts_hide_notes_from_student():
    secs = load_sections(TEX)
    p = {"typ": "wert", "frage": "F", "antwort": "1", "loesung": "x"}
    t = teacher_messages(secs[0], [p, p], 500)[0]["content"]
    assert secs[0]["text"][:200] in t and "höchstens 500".lower() in t.lower()
    s = student_messages(p, "ERKL")
    assert secs[0]["text"][:200] not in s[1]["content"] and s[1]["content"].startswith("Lernhilfe:\nERKL")
    assert student_messages(p)[1]["content"].startswith("Aufgabe:")
