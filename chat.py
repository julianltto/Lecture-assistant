import anthropic
from rag import retrieve

client = anthropic.Anthropic()
RULES = "You are the TA of this lecture. Answer only from the notes below. Cite pages as [p.X]."
MAX_TURNS = 4

messages = []
questions = []

def build_system_prompt(query):
    hits = retrieve(query)
    print("[retrieved:", [p["page"] for p in hits], "]")
    body = "\n\n".join(f'<page n="{p["page"]}">\n{p["text"]}\n</page>' for p in hits)
    return RULES + "\n\n" + body

while True:
    question = input("\nquestion> ").strip()
    if question == "quit":
        break
    if not question:
        continue

    questions.append(question)
    system = build_system_prompt(" ".join(questions[-3:]))
    messages.append({"role": "user", "content": question})
    messages = messages[-2 * MAX_TURNS + 1:]

    print("\nanswer> ", end="", flush=True)
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        msg = stream.get_final_message()
    print()

    answer = "".join(b.text for b in msg.content if b.type == "text")
    messages.append({"role": "assistant", "content": answer})

    print(f"[{msg.stop_reason}, in={msg.usage.input_tokens}, out={msg.usage.output_tokens}]")