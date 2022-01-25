from gptrun import gptrun
from functools import lru_cache

@gptrun
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


@gptrun(engine="text-curie-001")  # Use other engines for cost and/or precision
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


@gptrun
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
    


@gptrun(default=None)  # If you don't want to raise an exception if GPT3 returns nonsense
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


@gptrun  # Use '# doctest: +SKIP' on non-exact samples to avoid failing tests
def color(description):
    """
    Return the CSS color code by its description.

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


@lru_cache  # You can cache results to save some money
@gptrun
def weight(text):
    """
    Return the approximate weight of a well known person in kilograms.

    >>> weight("Megan Fox") # doctest: +SKIP
    53
    >>> weight("Anne Hathaway") # doctest: +SKIP
    59
    >>> weight("Amy Adams") # doctest: +SKIP
    52
    >>> weight("George Washington") # doctest: +SKIP
    87
    >>> weight("Harrison Ford") # doctest: +SKIP
    83
    """
    pass


@gptrun
def motto(character):
    """
    Returns an appropriate motto for the user.

    >>> motto("Goliath")  # doctest: +SKIP
    "Very tall, looking for a helmet"
    >>> motto("Superman")  # doctest: +SKIP
    "Harder, better, faster, stronger"
    >>> motto("Dumbo")  # doctest: +SKIP
    "Don't just fly, soar"
    """
    pass

    
if __name__ == '__main__':
    #
    # You can test your functions like so...
    #
    print("TESTING 'color':")
    color.test()

    #
    # You can use the functions as normal python functions:
    #
    if weight("Harrison Ford") > weight("Mahatma Gandhi"):
        print(f"No surprises here: {weight('Harrison Ford')=}, {weight('Mahatma Gandhi')=}")

    print(f'{capital("Australia")=}')
    print(f'{color("sun")=}')
    print(f'{spanish_rhyme("sillón")=}')
    print(f'{motto("nilp0inter")=}')
