import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore", message=".*y_pred contains classes.*")


def get_splits(protocol, batch):
    if protocol == "random":  # the WRONG protocol, kept as a reference
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for i, (tr, te) in enumerate(skf.split(X, y)):
            yield f"fold{i}", tr, te
    elif protocol == "batch1":  # train once on batch 1, age the model
        tr = np.where(batch == 1)[0]
        for k in range(2, 11):
            yield k, tr, np.where(batch == k)[0]
    elif protocol == "previous":  # train on the most recent batch only
        for k in range(2, 11):
            yield k, np.where(batch == k - 1)[0], np.where(batch == k)[0]
    elif protocol == "cumulative":  # train on everything before k
        for k in range(2, 11):
            yield k, np.where(batch < k)[0], np.where(batch == k)[0]


def run(X, y, model, splits):
    rows = []
    for test_id, tr, te in splits:
        m = model.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append(
            dict(
                test=test_id,
                acc=accuracy_score(y[te], p),
                bal_acc=balanced_accuracy_score(y[te], p),
            )
        )
    return pd.DataFrame(rows)
