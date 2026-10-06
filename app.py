import json, time, uuid
import streamlit as st
from core import stream_answer

FEEDBACK = "data/feedback.jsonl"

FAQ = """
**Which model answers my questions?**
Claude Opus 5 by Anthropic.

**Which courses are supported?**
Currently only *Linear Algebra*. If you have the LaTeX source (`.tex`) of your
course's lecture notes, email it to [s64lwu@uni-bonn.de] and we will add support for your course.

**Why can't I ask my next question?**
You need to rate the previous answer with 👍 or 👎 first. **Please always rate —
this is the most important signal we have for improving the answers.** The comment
box is optional, but very welcome when something is wrong or missing.

**Where do the answers come from?**
Only from the lecture notes. For each question the assistant looks up the most
relevant sections and answers from those. If it says the notes don't cover
something, try rephrasing with the terms used in the lecture.

**Can the answers be wrong?**
Yes. Always check important points against the lecture notes, and don't rely on it
for graded work.

**Which language should I use?**
English. The notes are in English and the search matches English terms best.

**Does it remember the conversation?**
It remembers the last few questions in this tab, so follow-up questions work.
Refreshing the page starts a new conversation.

**What data is stored?**
Your questions, the answers, your ratings and comments are saved to improve the
assistant. No name or login is recorded, but please don't enter personal information.
Questions are processed by Anthropic's API.
"""

st.title("Linear Algebra TA")
with st.popover("FAQ"):
    st.markdown(FAQ)

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.sid = uuid.uuid4().hex[:8] 

def save_feedback(i):
    h = st.session_state.history
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id": f"{st.session_state.sid}-{i}",
        "question": h[i - 1]["content"],
        "answer": h[i]["content"],
        "pages": h[i].get("pages", []),
        "rating": {0: "bad", 1: "good"}.get(st.session_state.get(f"rate_{i}")),
        "comment": st.session_state.get(f"comment_{i}", ""),
    }
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def feedback_widgets(i):
    st.feedback("thumbs", key=f"rate_{i}", on_change=save_feedback, args=(i,))
    st.text_input("comment", key=f"comment_{i}", placeholder="Comment (Optional)",
                  label_visibility="collapsed", on_change=save_feedback, args=(i,))

for i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            feedback_widgets(i)

last = len(st.session_state.history) - 1
need_rating = last >= 0 and st.session_state.get(f"rate_{last}") is None

if need_rating:
    st.caption("Please rate the last answer (👍 / 👎) before asking the next question.")

if question := st.chat_input("Ask about the lecture...", disabled=need_rating):
    with st.chat_message("user"):
        st.markdown(question)

    out = {}
    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(st.session_state.history, question, out))

    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "pages": out.get("pages", [])},
    ]
    st.rerun()
