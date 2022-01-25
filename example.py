from gptrun import gptrun
from functools import lru_cache


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
def color_by_description(description):
    """
    Return the CSS color code by its description.

    >>> color_by_description("black")
    "#000000"
    >>> color_by_description("white")
    "#ffffff"
    >>> color_by_description("deep blue ocean") # doctest: +SKIP
    "#003c5f"
    >>> color_by_description("the color of the sky") # doctest: +SKIP
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


#
# You can test your functions like so...
#
print("TESTING 'color_by_description':")
color_by_description.test()


#
# You can use the functions as normal python functions:
#
if weight("Harrison Ford") > weight("Mahatma Gandhi"):
    print(f"No surprises here: {weight('Harrison Ford')=}, {weight('Mahatma Gandhi')=}")

print(f'{capital("Australia")=}')
print(f'{color_by_description("nipples")=}')
