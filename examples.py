import json

from gptrun import chatgptrun
from functools import lru_cache

from gptrun.data import InvokationExample


@chatgptrun
def is_irony(text):
    """
    Returns True if text contains irony, False otherwise.

    >>> is_irony("I went to the movies last Friday; it was alright.")
    False
    >>> is_irony("He was one of the most supremely stupid men I have ever met. He taught me a great deal.")
    True
    >>> is_irony("Martha is at home right now.")
    False
    >>> is_irony("I didn't have time to write a short letter, so I wrote a long one instead.")
    True
    >>> is_irony("Silence is golden. Duct tape is silver.")
    True
    >>> is_irony("Since the 1930s, this company has worked alongside communities and clients of all sizes.")
    False
    """
    pass


@chatgptrun
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
    pass


@chatgptrun
def spanish_rhyme(word):
    """
    Return a rhyme given a word (in Spanish).

    >>> spanish_rhyme('cansado') # doctest: +SKIP
    'preparado'
    >>> spanish_rhyme('camión') # doctest: +SKIP
    'melón'
    >>> spanish_rhyme('adelantó') # doctest: +SKIP
    'suplantó'
    """
    pass
    


@chatgptrun(on_invalid_response=lambda:None)  # If you don't want to raise an exception if GPT3 returns nonsense
def sentiment(text):
    """
    Return the general sentiment of the phrase as 'positive', 'neutral' or 'negative'.

    >>> sentiment("I hate Mondays")
    "negative"
    >>> sentiment("I love my car")
    "positive"
    >>> sentiment("The sky is blue and so is the sea")
    "neutral"
    >>> sentiment("The sun is a star. There are lots of stars in the universe.")
    "neutral"
    """
    pass


@chatgptrun  # Use '# doctest: +SKIP' on non-exact samples to avoid failing tests
def color(description):
    """
    Return the lowercase CSS color code by its description.

    >>> color("black")
    "#000000"
    >>> color("white")
    "#ffffff"
    >>> color("deep blue ocean") # doctest: +SKIP
    "#003c5f"
    >>> color("the color of the sky") # doctest: +SKIP
    "#87ceeb"
    """
    pass


weight_examples = [
    InvokationExample.from_call("weight", "Megan Fox", _return=53),
    InvokationExample.from_call("weight", "Anne Hathaway", _return=59),
    InvokationExample.from_call("weight", "Amy Adams", _return=52),
    InvokationExample.from_call("weight", "George Washington", _return=87),
    InvokationExample.from_call("weight", "Harrison Ford", _return=83),
]

@chatgptrun(api_model="gpt-4")
def weight(text):
    """
    Return the approximate weight of a well known person or character in kilograms.

    If the weight is unknown return an approximation based on the height and gender.

    """
    pass


@chatgptrun(external_example_file="./motto.examples", num_examples=3)
def motto(character):
    """
    Returns an appropriate motto for the user.
    """
    pass

    
if __name__ == '__main__':
    #
    # You can test your functions like so...
    #
    print("TESTING 'color':")
    try:
        color.test_task_generalization()
    except Exception as e:
        print(e)

    #
    # You can use the functions as normal python functions:
    #
    wilson_fisk = weight("Wilson Fisk", _examples=weight_examples)
    mahatma_gandhi = weight("Mahatma Gandhi", _examples=weight_examples)
    if wilson_fisk > mahatma_gandhi:
        print(f"No surprises here: {wilson_fisk=}, {mahatma_gandhi=}")

    print(f'{capital("Australia")=}')
    print(f'{color("sun")=}')
    print(f'{spanish_rhyme("sillón")=}')
    print(f'{motto("nilp0inter")=}')

    # You can check the token for a given call
    print(f'{capital.calculate_tokens_per_call("France")=}')

    # You can also see the call details
    result = capital.call_with_details("Tunisia")
    print(json.dumps(result, indent=2))
