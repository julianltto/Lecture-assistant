from concurrent.futures import ThreadPoolExecutor

from la_teach.common import count_tokens, fit_student_messages, grade, strip_think, student_messages


class StudentClient:
    def __init__(self, base_url, model, n=1, temperature=0.7, top_p=0.8, max_tokens=2048,
                 thinking=False, workers=64, timeout=600, max_context=12288, tokenizer=None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=3)
        self.model = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_context = max_context
        self.tok = tokenizer
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"requests": 0, "failed": 0, "ctx_truncated": 0, "length_stops": 0, "answers": 0}

    def prompt_tokens(self, messages):
        return count_tokens(self.tok, messages, self.thinking)

    def fit(self, problem, ctx, label):
        if self.tok is None:
            return student_messages(problem, ctx, label)
        messages, truncated = fit_student_messages(self.tok, problem, ctx, label,
                                                   self.max_context - self.max_tokens, self.thinking)
        self.stats["ctx_truncated"] += truncated
        return messages

    def _ask(self, messages):
        from openai import BadRequestError

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
            return [(c.message.content or "", c.finish_reason) for c in r.choices], False
        except BadRequestError:
            raise
        except Exception as e:
            print(f"[student] request failed: {e!r}", flush=True)
            return [("", "error")] * self.n, True

    def accuracy(self, jobs):
        msgs = [self.fit(p, ctx, label) for p, ctx, label in jobs]
        futures = [self.pool.submit(self._ask, m) for m in msgs]
        accs = []
        for (p, _, _), f in zip(jobs, futures):
            outs, failed = f.result()
            self.stats["requests"] += 1
            self.stats["failed"] += failed
            self.stats["answers"] += len(outs)
            self.stats["length_stops"] += sum(fr == "length" for _, fr in outs)
            accs.append(sum(grade(p, strip_think(o)) for o, _ in outs) / len(outs))
        return accs
