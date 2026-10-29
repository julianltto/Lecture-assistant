# This Week: Linear Algebra TA Website

Site: https://34-107-114-129.sslip.io/

## What it does

A teaching assistant for the *Linear Algebra* lecture. Students ask questions, and an LLM (GPT-6 Luna) answers only from the lecture notes (Blomer's German script) and cites the sections as `[sec.X]`. The goal is to collect student feedback and use it to train a small assistant that runs on a laptop.

## Features

- Chat interface that supports follow-up questions.
- For each question, the most relevant sections of the notes are retrieved and the answer is based on them.
- Students must rate every answer 👍/👎 before they can ask the next question.
- After a 👎, students tick what was wrong (e.g. mathematically wrong, missing steps, wrong citation) and can say how the answer should be improved (e.g. step by step, a worked example, the intuition first). They can also leave a comment or skip.
- An FAQ explains the project and how the data is used.

## Data collected

For each answer:
- the question and the earlier questions in the conversation
- the answer and the lecture sections it used
- the rating, the ticked problems and improvements, and the comment
- a timestamp and a random session ID

Currently, the model runs on Google Cloud, will move it to uni's server asap.

# Some future ideas

- The original plan was for my small, specialized model to beat frontier models at explaining math. My latest tests show that frontier models already explain math very well, so there is no real room to beat them there.
- Discuss with the professor which weakness that the current frontier models still have (e.g. should not try to reinterpret the question, but rather should point out the misunderstanding).
- Try to run a workshop to get students involved.

