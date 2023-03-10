from dataclasses import dataclass
import ast
import doctest
import functools
import inspect
import inspect
import os
import random
import textwrap

import openai
import tiktoken


__all__ = ['gpt3run', 'RAISE_EXCEPTION']

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class InvokationExample:
    source: str
    call_args: list[object]
    call_kwargs: dict[str, object]
    want: str
    options: dict

    @classmethod
    def from_doctest_example(cls, example):
        source = ast.parse(example.source).body[0]
        if not isinstance(source.value, ast.Call):
            raise ValueError("Examples must be function calls only!")
        want = ast.parse(example.want).body[0]
        call_args = [ast.unparse(x) for x in source.value.args]
        call_kwargs = {kw.arg: ast.unparse(kw.value) for kw in source.value.keywords}
        return cls(source=ast.unparse(source),
                   call_args=call_args,
                   call_kwargs=call_kwargs,
                   want=ast.unparse(want),
                   options=example.options)


@dataclass
class FakeFunctionDefinition:
    summary: str
    examples: list[InvokationExample]
    
    @staticmethod
    def get_summary(docstring):
        """Return the first paragraph of a function's docstring. """
        dedented_docstring = textwrap.dedent(docstring)
        paragraphs = dedented_docstring.strip().split('\n\n')
        first_paragraph = paragraphs[0].strip() if paragraphs else ''
        return ' '.join((l.strip() for l in first_paragraph.splitlines()))

    @classmethod
    def from_docstring(cls, docstring, external_examples=None):
        """Initialize a FakeFunctionDefinition from a docstring."""
        summary = cls.get_summary(docstring)
        if external_examples:
            examples = doctest.DocTestParser().get_examples(external_examples)
        else:
            examples = doctest.DocTestParser().get_examples(docstring)

        return cls(summary=summary,
                   examples=[InvokationExample.from_doctest_example(e) for e in examples])


def gpt3run(*args, **kwargs):
    """
    A decorator that transform a function without code but with a docstring
    containing examples into a function that calls OpenAI's generation API and
    perform few shot prompting on the examples.

    :param f: The function to transform.
    :param on_failure: A function to call if GPT3 fails to return a valid output.
    :param engine: The OpenAI engine to use.
    :param example_filepath: A path to a file containing external examples instead of using the docstring.
    :param num_examples: The number of examples to use. If 0, all examples are used.
    :param completion_kwargs: Additional keyword arguments to pass to the OpenAI API.

    """
    if kwargs:
        def gpt3run(f):
            return functools.wraps(f)(CompletionAPIRunner(f, **kwargs))
        return gpt3run
    else:
        return functools.wraps(args[0])(CompletionAPIRunner(args[0]))


def RAISE_EXCEPTION():
    raise ValueError("GPT returned bad output")


class CompletionAPIRunner:
    """Infere call result using OpenAI's completion API."""
    def __init__(self, f,
                 engine="text-davinci-003",
                 on_failure=RAISE_EXCEPTION,
                 example_filepath=None,
                 num_examples=0,
                 **completion_kwargs):
        """See `gpt3run` decorator."""
        self.name = f.__name__

        self.engine = engine
        self.on_failure = on_failure

        # Examples can be provided from an external `example_filepath` or as a docstring body.
        examples = None
        if example_filepath is not None:
            with open(example_filepath) as example_file:
                examples = example_file.read()

        self.definition = FakeFunctionDefinition.from_docstring(f.__doc__, external_examples=examples)

        self.num_examples = min(num_examples or len(self.definition.examples), len(self.definition.examples))  # Cap to the number of examples
        self.completion_kwargs = completion_kwargs

    def calculate_tokens_per_call(self, *args, **kwargs):
        tokenizer = tiktoken.encoding_for_model(self.engine)
        if self.num_examples == len(self.definition.examples):  # In this case we can provide an exact answer 
            return {"result_type": "exact",
                    "value": len(tokenizer.encode(self.make_prompt(*args, **kwargs)))}
        else: # We only can approximate the number of tokens per call by sampling
            return {"result_type": "average",
                    "value": sum(len(tokenizer.encode(self.make_prompt(*args, **kwargs))) for _ in range(1000)) / 1000}

    def make_prompt(self, *args, _examples=None, **kwargs):
        """Build the prompt for the given set of parameters."""

        doc = f'>>> {self.name}.__doc__\n{self.definition.summary!r}'

        example_base = self.definition.examples if _examples is None else _examples
        num_examples = min(self.num_examples, len(example_base))
        examples = "\n".join(f'>>> {e.source}\n{e.want}' for e in random.sample(example_base, k=num_examples))

        args = [repr(a) for a in args]
        kwargs = [f'{k}={v!r}' for k, v in kwargs.items()]
        call = f'>>> {self.name}({", ".join(args + kwargs)})'

        return '\n'.join([doc, examples, call])

    def __call__(self, *args, **kwargs):
        response = openai.Completion.create(
          engine=self.engine,
          prompt = self.make_prompt(*args, **kwargs),
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

    def _make_test_prompts(self):
        for i in range(len(self.definition.examples)):
            preamble = self.definition.examples[:i] + self.definition.examples[i+1:]
            missing = self.definition.examples[i]
            if missing.options.get(doctest.SKIP, None):
                continue
            yield (self.make_prompt(*missing.call_args, _examples=preamble, **missing.call_kwargs), missing.want)
    
    def test_examples(self):
        """
        This function let you test the ability to generalize the task with the
        examples given in the docstring.

        This works by prompting the model as many times as examples are in the
        definition, plucking out one example at a time and testing if that call
        returns the expected output for that example.

        """
        for i, (prompt, wanted) in enumerate(self._make_test_prompts()):
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
