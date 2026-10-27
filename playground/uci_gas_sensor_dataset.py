import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file


class UCIGasSensorDataset:
    GASES = {
        0: "ethanol",
        1: "ethylene",
        2: "ammonia",
        3: "acetaldehyde",
        4: "acetone",
        5: "toluene",
    }

    def __init__(self):
        self.df = None
        self.feature_columns = None

    def load(self):
        names = [
            "dR",
            "dR_norm",
            "ema_up_0.001",
            "ema_up_0.01",
            "ema_up_0.1",
            "ema_down_0.001",
            "ema_down_0.01",
            "ema_down_0.1",
        ]
        columns = [f"s{s+1}_{names[k]}" for s in range(16) for k in range(8)]

        frames = []
        for b in range(1, 11):
            X, y = load_svmlight_file(
                f"data/batch{b}.dat", n_features=128, zero_based=False
            )
            df = pd.DataFrame(X.toarray(), columns=columns)
            df["gas"] = y.astype(int) - 1 # Make zero-based
            df["batch"] = b
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)
        self.df["gas_name"] = self.df["gas"].map(UCIGasSensorDataset.GASES)

        self.feature_columns = list(self.df.filter(regex=r"^s\d+_").columns)
