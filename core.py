import anthropic
from rag import retrieve

model = "claude-opus-5"
client = anthropic.Anthropic()
RULES = ("You are the TA of this lecture. Answer only from the notes below. "
         "Cite pages as [sec.X]. If the notes don't cover it, say so. "
         "Write math as $...$ or $$...$$. "
         "The notes use custom macros that the display cannot render. NEVER output them; translate them if needed.")
MAX_TURNS = 4

def build_system_prompt(query):
    hits = retrieve(query)
    if not hits:
        return RULES + "\n\nNo section of the lecture notes matches this question.", hits
    body = "\n\n".join(f'<page n="{p["page"]}">\n{p["text"]}\n</page>' for p in hits)
    return RULES + "\n\n" + body, hits

def stream_answer(history, question, out):
    messages = (history + [{"role": "user", "content": question}])[-2 * MAX_TURNS + 1:]
    messages = [{"role": m["role"], "content": m["content"]} for m in messages]  # API 只收这两个字段
    questions = [m["content"] for m in messages if m["role"] == "user"]
    system, hits = build_system_prompt(" ".join(questions[-3:]))
    out["pages"] = [p["page"] for p in hits]

    with client.messages.stream(
        model=model,
        max_tokens=1000,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
        out["message"] = stream.get_final_message()
