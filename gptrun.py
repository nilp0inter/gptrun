import ast
import doctest
import functools
import inspect
import os
import random

import openai


__all__ = ['gptrun', 'RAISE_EXCEPTION']

openai.api_key = os.getenv("OPENAI_API_KEY")


def gptrun(*args, **kwargs):
    """
    A decorator that transform a function without code but with a docstring
    containing examples into a function that calls OpenAI's API and perform few
    shot prompting on the examples.

    :param f: The function to transform.
    :param on_failure: A function to call if GPT3 fails to return a valid output.
    :param engine: The OpenAI engine to use.
    :param example_filepath: A path to a file containing external examples instead of using the docstring.
    :param num_examples: The number of examples to use. If 0, all examples are used.
    :param completion_kwargs: Additional keyword arguments to pass to the OpenAI API.

    """
    if kwargs:
        def gptrun(f):
            return functools.wraps(f)(GPTRunner(f, **kwargs))
        return gptrun
    else:
        return functools.wraps(args[0])(GPTRunner(args[0]))


def RAISE_EXCEPTION():
    raise ValueError("GPT3 returned bad output")


class GPTRunner:
    def __init__(self, f,
                 engine="text-davinci-001",
                 on_failure=RAISE_EXCEPTION,
                 example_filepath=None,
                 num_examples=0,
                 **completion_kwargs):
        """See `gptrun` decorator."""
        self.name = f.__name__
        self.summary = inspect.getdoc(f).splitlines()[0]

        self.engine = engine
        self.on_failure = on_failure

        # Examples can be provided from an external `example_filepath` or as a docstring body.
        examples = ""
        if example_filepath is not None:
            with open(example_filepath) as example_file:
                examples = example_file.read()
        else:
            examples = f.__doc__
        self.examples = doctest.DocTestParser().get_examples(examples)

        self.num_examples = min(num_examples or len(self.examples), len(self.examples))  # Cap to the number of examples
        self.completion_kwargs = completion_kwargs

    def make_prompt(self, *args, **kwargs):
        args = [repr(a) for a in args]
        kwargs = [f'{k}={v!r}' for k, v in kwargs.items()]
        call = f'>>> {self.name}({", ".join(args + kwargs)})'
        examples = "\n".join(f'>>> {ast.unparse(ast.parse(e.source))}\n{ast.unparse(ast.parse(e.want))}' for e in random.sample(self.examples, k=self.num_examples))
        doc = f'>>> {self.name}.__doc__\n{self.summary!r}'
        return '\n'.join([doc, examples, call])

    def __call__(self, *args, **kwargs):
        prompt = self.make_prompt(*args, **kwargs)
        response = openai.Completion.create(
          engine=self.engine,
          prompt=prompt,
          top_p=1,
          **self.completion_kwargs
        )
        try:
            return ast.literal_eval(response['choices'][0]['text'].strip())
        except Exception as gpt_exception:
            try:
                return self.on_failure()
            except Exception as user_exception:
                raise user_exception from gpt_exception

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
              top_p=1,
              **self.completion_kwargs
            )
            try:
                current = ast.literal_eval(response['choices'][0]['text'].strip())
            except Exception:
                assert False, "GPT3 returned bad output"
                
            wanted = ast.literal_eval(wanted)
            assert current == wanted, f'In test #{i}: {prompt}, and got {current!r} instead of {wanted!r}'
            print('.', end='')
        print('')
