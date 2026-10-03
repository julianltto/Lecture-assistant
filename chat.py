import anthropic


notes = open("data/lecture_notes/Linear_Algebra.tex", encoding="utf-8").read()
client = anthropic.Anthropic()
RULES = "You are the TA of this lecture. Answer only from the notes below. Cite pages as [p.X]."
system = [
    {"type": "text", "text": RULES},
    {"type": "text", "text": "<lecture>\n" + notes + "\n<lecture>"},
]
messages = []

while True:
    question = input("\nquestion> ".strip())
    if question == "quit":
        break

    messages.append({"role": "user", "content": question})
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