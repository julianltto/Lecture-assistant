import json, os, time, uuid
import streamlit as st
from core import stream_answer, model, model_label

FEEDBACK_DIR = "data/feedback" 

ISSUES = ["Mathematically wrong", "Doesn't answer my question", "Wrong or missing citation",
          "Says the notes don't cover it, but they do", "Hard to understand",
          "Too long", "Too short / missing steps", "Math or formatting broken"]
IMPROVE = ["Explain step by step", "Add a worked example", "Give the intuition first",
           "Use the lecture's notation", "Be shorter", "Include the proof"]

FAQ = f"""
:blue[**What is this?**]
A project to build a lecture assistant for students. You ask questions about the
*Linear Algebra* lecture, a large language model (LLM) answers from the lecture notes
(citing them as [sec.X]), and you tell us whether the answer helped and, if not, what
was wrong and how it should be better. At this stage we collect that feedback and use
it to train a lecture assistant that is small enough to run on a laptop and that will
be made available to all students of the University of Bonn. We will also experiment
with different ways for students to contribute to this project. **Your feedback is
very welcome!**

:blue[**Which model answers my questions?**]
{model_label}.

:blue[**Which courses are supported?**]
Currently only *Linear Algebra*. If you have the LaTeX source (`.tex`) of your
course's lecture notes, email it to [s64lwu@uni-bonn.de] and we will add support for your course.

:blue[**Why can't I ask my next question?**]
You need to rate the previous answer with 👍 or 👎 first. **Please always rate —
this is the most important signal we have for improving the answers.** After a 👎,
also tick at least one thing that was wrong. How it should be improved and the
comment box are optional, but very welcome.

:blue[**Where do the answers come from?**]
Only from the lecture notes. For each question the assistant looks up the most
relevant sections and answers from those. If it says the notes don't cover
something, try rephrasing with the terms used in the lecture.

:blue[**Can the answers be wrong?**]
Yes. Always check important points against the lecture notes, and don't rely on it
for graded work.

:blue[**Which language should I use?**]
German. Language using needs to match the language of the lecture note.

:blue[**Does it remember the conversation?**]
It remembers the last few questions in this tab, so follow-up questions work.
Refreshing the page starts a new conversation.

:blue[**What data is stored?**]
For each answer we save:
- your question and the earlier questions of this conversation,
- the answer and the lecture sections it was based on,
- your rating, the problems and improvements you ticked, and your comment,
- the time of your rating and a random session ID.

This data is used to train the lecture assistant. No name or login is recorded, so
please don't enter personal information. Your IP address is used to deliver the
website but is not stored: the server keeps no access logs. Questions are processed
by OpenAI's API.
"""

st.title("Linear Algebra TA")
with st.popover("FAQ"):
    st.markdown(FAQ)

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.sid = uuid.uuid4().hex[:8] 

def save_feedback(i):
    h = st.session_state.history
    bad = st.session_state.get(f"rate_{i}") == 0
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id": f"{st.session_state.sid}-{i}",
        "model": h[i].get("model"),
        "question": h[i - 1]["content"],
        "answer": h[i]["content"],
        "pages": h[i].get("pages", []),
        "request": h[i].get("request"),
        "rating": {0: "bad", 1: "good"}.get(st.session_state.get(f"rate_{i}")),
        "issues": (st.session_state.get(f"issues_{i}") or []) if bad else [],
        "improve": (st.session_state.get(f"improve_{i}") or []) if bad else [],
        "comment": st.session_state.get(f"comment_{i}", ""),
    }
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    path = os.path.join(FEEDBACK_DIR, record["id"] + ".json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path) 

def feedback_widgets(i):
    st.feedback("thumbs", key=f"rate_{i}", on_change=save_feedback, args=(i,))
    if st.session_state.get(f"rate_{i}") == 0:
        st.pills("What was wrong?", ISSUES, selection_mode="multi", key=f"issues_{i}",
                 on_change=save_feedback, args=(i,))
        st.pills("How should it be improved?", IMPROVE, selection_mode="multi", key=f"improve_{i}",
                 on_change=save_feedback, args=(i,))
    st.text_input("comment", key=f"comment_{i}", placeholder="Comment (Optional)",
                  label_visibility="collapsed", on_change=save_feedback, args=(i,))

for i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            feedback_widgets(i)

last = len(st.session_state.history) - 1
need_rating = last >= 0 and st.session_state.get(f"rate_{last}") is None
need_reason = (last >= 0 and st.session_state.get(f"rate_{last}") == 0
               and not st.session_state.get(f"issues_{last}"))

if need_rating:
    st.caption("Please rate the last answer (👍 / 👎) before asking the next question.")
elif need_reason:
    st.caption("Please select what was wrong with the last answer before asking the next question.")

if question := st.chat_input("Ask about the lecture...", disabled=need_rating or need_reason):
    with st.chat_message("user"):
        st.markdown(question)

    out = {}
    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(st.session_state.history, question, out))

    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "pages": out.get("pages", []), "model": model,
         "request": out.get("request")},
    ]
    st.rerun()
