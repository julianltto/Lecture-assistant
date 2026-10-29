from concurrent.futures import ThreadPoolExecutor

from la_teach.common import grade, strip_think, student_messages


class StudentClient:
    def __init__(self, base_url, model, n=1, temperature=0.7, top_p=0.8, max_tokens=2048,
                 thinking=False, workers=64, timeout=600):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=3)
        self.model = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def _ask(self, messages):
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=self.n,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
            )
            return [c.message.content or "" for c in r.choices]
        except Exception as e:
            print(f"[student] request failed: {e!r}", flush=True)
            return [""] * self.n

    def accuracy(self, jobs):
        futures = [self.pool.submit(self._ask, student_messages(p, ctx, label)) for p, ctx, label in jobs]
        accs = []
        for (p, _, _), f in zip(jobs, futures):
            outs = f.result()
            accs.append(sum(grade(p, strip_think(o)) for o in outs) / len(outs))
        return accs
