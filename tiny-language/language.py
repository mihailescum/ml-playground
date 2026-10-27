import numpy as np

vocabulary = [
    "a",
    "b",
    "c",
    "-",
    "@",
]  # "-" is the end token while "@" is the start token
priors = {
    vocabulary[0]: (
        lambda x: int(x * 3),  # distribution for vocabulary is uniform
        lambda x: 1 + int(5 * x),  # distribution for repetitions
    ),
    vocabulary[1]: (
        lambda x: 1,  # always maps to second letter
        lambda x: 2 if x < 0.5 else 4,  # always eiter 2 or 4 letters
    ),
    vocabulary[2]: (
        lambda x: (
            0 if x < 0.3 else (1 if x < 0.5 else 2)
        ),  # skewed between first and third letter
        lambda x: min(5, int(1 / x)),
    ),
}


def generate(symbol):
    r = np.random.uniform(0, 1, size=2)
    return priors[symbol][1](r[1]) * vocabulary[priors[symbol][0](r[0])]
