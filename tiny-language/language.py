import numpy as np
import torch


class TinyLanguageDataset(torch.utils.data.IterableDataset):
    vocabulary = [
        "a",
        "b",
        "c",
        "-",
        "@",
    ]  # "@" is the end token while "-" is the start token
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

    def __init__(self, min_seq_length=1, max_seq_length=20, epoch_size=1000):
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            l = np.random.randint(self.min_seq_length, self.max_seq_length)
            x = TinyLanguageDataset.to_string(np.random.randint(0, 3, size=l))
            yield x, TinyLanguageDataset.sample(x)

    def to_string(tokenized):
        return "".join([TinyLanguageDataset.vocabulary[i] for i in tokenized])

    def to_tokens(string):
        return torch.tensor(
            [TinyLanguageDataset.vocabulary.index(letter) for letter in string]
        )

    def sample(input):
        r = np.random.uniform(0, 1, size=(len(input), 2))
        return "".join(
            (
                TinyLanguageDataset.priors[symbol][1](r[i, 1])
                * TinyLanguageDataset.vocabulary[
                    TinyLanguageDataset.priors[symbol][0](r[i, 0])
                ]
            )
            for i, symbol in enumerate(input)
        )
