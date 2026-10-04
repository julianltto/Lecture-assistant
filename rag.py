import json, re
from rank_bm25 import BM25Okapi

pages = [json.loads(i) for i in open("data/chunks.jsonl", encoding="utf-8")]

STOP = set("""a an the is are was be of to in on for and or not with by as at from it this that
these those what which who how why when where do does did i you we can could should would
me my your if then there their its into about than so""".split())

def tokenize(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

bm25 = BM25Okapi([tokenize(p["text"]) for p in pages])

def retrieve(query, k=3):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(pages)), key=lambda i: -scores[i])[:k]
    top = [i for i in top if scores[i] > 0]
    return [pages[i] for i in sorted(top)]