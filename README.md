# gptrun
Don't feel like coding today?  Use the power of GPT3 to execute any function inside your programs just by giving some doctests.

**How is this different from Copilot? Is that what you said?** Copilot generates code that you run.  This beauty uses GPT3 to compute the answer to each function call.

## Installation
```console
$ pip install git+https://github.com/nilp0inter/gptrun@main
```

## Example

First you need an OPENAI API key. Get it here: https://openai.com/api/

```console
$ export OPENAI_API_KEY="<your OPENAI key>"
```

A code sample:

```python
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
    pass  # I don't feel like coding today (:


>>> capital.test()  # You can test your "code".  Don't let them blame you on coverage.
...

>>> capital("China")
"Beijing"

```

## Not impressed yet? 🤔

```python
>>> from examples import is_irony
>>> is_irony("If you find me offensive. Then I suggest you quit finding me.")
True
>>> is_irony("If you find me offensive. Then I suggest you quit.")
False

```

Look other examples in `examples.py`. And if you came up with more send me a pull request.

You can adjust GPT3 parameters using the decorator. See `examples.py`.


This code is production ready and 💯% certified by the Ministry of Silly Walks.
