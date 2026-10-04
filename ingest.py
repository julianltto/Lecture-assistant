import json, re

raw = open("data/lecture_notes/Linear_Algebra.tex", encoding="utf-8").read()
raw = raw.split(r"\begin{document}", 1)[-1].split(r"\end{document}", 1)[0] 
chunks = re.split(r"(?=\\(?:chapter|section)\*?\{)", raw)

merged = []
for c in chunks:
    if merged and len(merged[-1].strip()) < 100:
        merged[-1] += c
    else:
        merged.append(c)

with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for i, text in enumerate(merged, 1):
        if text.strip():
            f.write(json.dumps({"page": i, "text": text.strip()}, ensure_ascii=False) + "\n")
print(len(merged), "chunks")
