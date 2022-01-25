import doctest
import ast
import inspect
import os
import functools

import openai


__all__ = ['gptrun', 'RAISE_EXCEPTION']

openai.api_key = os.getenv("OPENAI_API_KEY")


def gptrun(*args, **kwargs):
    if kwargs:
        def gptrun(f):
            return functools.wraps(f)(GPTRunner(f, **kwargs))
        return gptrun
    else:
        return functools.wraps(args[0])(GPTRunner(args[0]))


RAISE_EXCEPTION = object()

class GPTRunner:
    def __init__(self, f, default=RAISE_EXCEPTION, engine="text-davinci-001", temperature=0.0, max_tokens=64, frequency_penalty=0, presence_penalty=0):
        self.name = f.__name__
        self.summary = inspect.getdoc(f).splitlines()[0]
        self.examples = doctest.DocTestParser().get_examples(f.__doc__)
        self.default = default

        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def __call__(self, *args, **kwargs):
        args = [repr(a) for a in args]
        kwargs = [f'{k}={v!r}' for k, v in kwargs.items()]
        call = f'>>> {self.name}({", ".join(args+kwargs)})'
        examples = "".join(f'>>> {e.source.split("#")[0]}{e.want}' for e in self.examples)
        prompt = examples + call
        response = openai.Completion.create(
          engine=self.engine,
          prompt=prompt,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          top_p=1,
          frequency_penalty=self.frequency_penalty,
          presence_penalty=self.presence_penalty
        )
        try:
            return ast.literal_eval(response['choices'][0]['text'].strip())
        except Exception as exc:
            if self.default is RAISE_EXCEPTION:
                raise ValueError("GPT3 returned bad output") from exc
            else:
                return self.default

    def _get_tests(self):
        for i in range(len(self.examples)):
            preamble = self.examples[:i] + self.examples[i+1:]
            missing = self.examples[i]
            if missing.options.get(doctest.SKIP, None):
                continue
            examples = "".join([f'>>> {p.source.split("#")[0]}{p.want}' for p in preamble]) + f'>>> {missing.source}'
            yield (f'>>> {self.name}.__doc__\n{self.summary!r}\n{examples}', missing.want.rstrip('\n'))
    
    def test(self):
        for i, (prompt, wanted) in enumerate(self._get_tests()):
            response = openai.Completion.create(
              engine=self.engine,
              prompt=prompt,
              temperature=self.temperature,
              max_tokens=self.max_tokens,
              top_p=1,
              frequency_penalty=self.frequency_penalty,
              presence_penalty=self.presence_penalty
            )
            try:
                current = ast.literal_eval(response['choices'][0]['text'].strip())
            except Exception:
                assert False, "GPT3 returned bad output"
                
            wanted = ast.literal_eval(wanted)
            assert current == wanted, f'In test #{i}: {prompt}, and got {current!r} instead of {wanted!r}'
            print('.', end='')
        print('')
