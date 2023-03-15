# gptrun

gptrun is a Python library that enables you to seamlessly harness the power of
language models like GPT-3 and ChatGPT for rapid prototyping without writing
any code, using a technique called few-shot prompting.

Instead of generating code to run like Github's Copilot, `gptrun` directly
computes the answers to your function calls using GPT-3. All you need to do is
provide some doctests for your desired function, and let GPT-3 do the rest.

## Features

- Effortless function mocking using GPT-3
- Test-driven development with doctests
- Customizable GPT-3 parameters using decorators
- Easy integration with other AI models

## Installation

To install `gptrun`, follow these steps:

1. Install the library from the GitHub repository:

```console
$ pip install git+https://github.com/nilp0inter/gptrun@main
```

2. Set up your OpenAI API key:

```console
$ export OPENAI_API_KEY="<your OPENAI key>"
```
If you don't have an API key, you can obtain one from the OpenAI API website.

## Usage

Using gptrun is as simple as adding a decorator to your functions and providing
some doctests. Here's a basic example:

```python

from gptrun import gptrun

@gptrun
def capital(country):
    """
    Return the capital of a country.

    >>> capital("Angola")
    "Luanda"
    >>> capital("France")
    "Paris"
    >>> capital("Spain")
    "Madrid"
    """
    pass  # No need to write any code!

# Test your function
capital.test_task_generalization()

# Call your function
print(capital("China"))  # Output: "Beijing"
```
For more advanced usage and customization, check out the examples in `examples.py`.

## Contributing

Contributions to gptrun are always welcome! If you have an idea for a new
feature, a bug report, or a question, please open an issue on GitHub. To submit
a pull request, please fork the repository and create a new branch with your
changes.

