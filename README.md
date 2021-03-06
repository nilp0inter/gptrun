# gptrun
Don't feel like coding today?  Use the power of GPT3 to imagine the result of any function. You only need some doctests.

**How is this different from Github's Copilot? Is that what you're thinking?** Copilot generates code that you run.  This beauty uses GPT3 to compute the answer to each function call.

## Example

First you need an OPENAI API key. Get yours here: https://openai.com/api/

```console
$ pip install git+https://github.com/nilp0inter/gptrun@main
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

Look other examples in `examples.py`. And if you came up with more, just send me a pull request.

You can adjust GPT3 parameters using the decorator. See `examples.py`.

## How is this useful?

This can be useful for rapid prototyping of AI based systems.  Instead of developing your own model, just mock it with GPT3.


This code is production ready and 💯% certified by the Ministry of Silly Walks.
