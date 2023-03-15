from abc import ABC, abstractmethod
from itertools import chain
import ast
import doctest
import functools
import os
import random
import sys

import openai
import tiktoken

from .data import FakeFunctionDefinition, InvokationExample


__all__ = ['gpt3run', 'chatgptrun', 'RAISE_EXCEPTION']

openai.api_key = os.getenv("OPENAI_API_KEY")


class Runner(ABC):
    @abstractmethod
    def calculate_tokens_per_call(self, *args, **kwargs):
        """
        Return the number of tokens per call.

        Depending on the runner, this can be an exact value or an average value.

        Example: {"result_type": "exact", "value": 100}

        """
        pass

    @abstractmethod
    def make_prompt(self, *args, **kwargs):
        """
        Return the prompt to use for the API.

        This prompt will be used to generate the output of the function.

        """
        pass

    @abstractmethod
    def call_api(self, *args, **kwargs):
        """
        Call the API with the given parameters returning the raw response.

        """
        pass

    @abstractmethod
    def api_response_to_text(self, response):
        """
        Return the part of the response representing the model text completion.

        """
        pass


    @abstractmethod
    def test_examples(self):
        """
        This function let you test the ability to generalize the task with the
        examples given in the docstring.

        This works by prompting the model as many times as examples are in the
        definition, plucking out one example at a time and testing if that call
        returns the expected output for that example.

        """
        pass

    def __call__(self, *args, **kwargs):
        """Run the runner."""

        response = self.call_api(*args, **kwargs)
        completion = self.api_response_to_text(response)

        try:
            return ast.literal_eval(completion)
        except Exception as gpt_exception:
            try:
                return self.on_failure()
            except Exception as user_exception:
                raise user_exception from gpt_exception


def RAISE_EXCEPTION():
    raise ValueError("GPT returned bad output")


class CompletionAPIRunner(Runner):
    """Infere call result using OpenAI's completion API."""
    def __init__(self, f,
                 engine="text-davinci-003",
                 on_failure=RAISE_EXCEPTION,
                 example_filepath=None,
                 num_examples=0,
                 **api_kwargs):
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
        self.api_kwargs = api_kwargs

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

    def call_api(self, *args, **kwargs):
        return openai.Completion.create(
          engine=self.engine,
          prompt = self.make_prompt(*args, **kwargs),
          top_p=1,
          **self.api_kwargs
        )

    def api_response_to_text(self, response):
        return response['choices'][0]['text'].strip()

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
              **self.api_kwargs
            )
            try:
                current = ast.literal_eval(response['choices'][0]['text'].strip())
            except Exception:
                assert False, "GPT3 returned bad output"
                
            wanted = ast.literal_eval(wanted)
            assert current == wanted, f'In test #{i}: {prompt}, and got {current!r} instead of {wanted!r}'
            print('.', end='')
        print('')


class ChatCompletionAPIRunner(Runner):
    """Infere call result using OpenAI's chat API."""
    def __init__(self, f,
                 engine="gpt-3.5-turbo",
                 on_failure=RAISE_EXCEPTION,
                 example_filepath=None,
                 num_examples=0,
                 **api_kwargs):
        """See `chatgptrun` decorator."""
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
        self.api_kwargs = api_kwargs

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

        python_prompt = [{"role": "system", "content": f'Python {sys.version} (main, Feb  7 2023, 12:19:31) [GCC 12.2.0] on {sys.platform}\nType "help", "copyright", "credits" or "license" for more information.'}]

        doc = [{"role": "user", "content": f'>>> {self.name}.__doc__'},
               {"role": "assistant", "content": f'{self.definition.summary!r}'}]

        example_base = self.definition.examples if _examples is None else _examples
        num_examples = min(self.num_examples, len(example_base))
        examples = [({"role": "user", "content": f'>>> {e.source}'},
                     {"role": "assistant", "content": f'{e.want}'})
                    for e in random.sample(example_base, k=num_examples)]
        examples = list(chain.from_iterable(examples))

        args = [repr(a) for a in args]
        kwargs = [f'{k}={v!r}' for k, v in kwargs.items()]
        call = [{"role": "user", "content": f'>>> {self.name}({", ".join(args + kwargs)})'}]

        return python_prompt + doc + examples + call

    def call_api(self, *args, **kwargs):
        return openai.ChatCompletion.create(
          model=self.engine,
          messages=self.make_prompt(*args, **kwargs),
          **self.api_kwargs
        )

    def api_response_to_text(self, response):
        return response['choices'][0]['message']['content'].strip()

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
            response = openai.ChatCompletion.create(
              model=self.engine,
              messages=prompt,
              **self.api_kwargs
            )
            try:
                current = ast.literal_eval(response['choices'][0]['message']['content'].strip())
            except Exception:
                assert False, "GPT3 returned bad output"
                
            wanted = ast.literal_eval(wanted)
            assert current == wanted, f'In test #{i}: {prompt}, and got {current!r} instead of {wanted!r}'
            print('.', end='')
        print('')


def _make_runner_decorator(runner):
    """Make a decorator that transform a function into a runner."""
    def runner_decorator(*args, **kwargs):
        """
        A decorator that transform a function without code but with a docstring
        containing examples into a function that calls some OpenAI API and
        perform few shot prompting on the examples.

        :param f: The function to transform.
        :param on_failure: A function to call if the model fails to return a valid Python output.
        :param engine: The OpenAI engine to use.
        :param example_filepath: A path to a file containing external examples instead of using the docstring.
        :param num_examples: The number of examples to use. If 0, all examples are used.
        :param api_kwargs: Additional keyword arguments to pass to the OpenAI API.

        """
        if kwargs:
            def wrapper(f):
                return functools.wraps(f)(runner(f, **kwargs))
            return wrapper
        else:
            return functools.wraps(args[0])(runner(args[0]))

    return runner_decorator


gpt3run = _make_runner_decorator(CompletionAPIRunner)
chatgptrun = _make_runner_decorator(ChatCompletionAPIRunner)
