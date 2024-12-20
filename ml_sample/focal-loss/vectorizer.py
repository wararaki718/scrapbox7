import numpy as np
import pandas as pd


class Vectorizer:
    def transform(self, df: pd.DataFrame, eps: float=0.001) -> np.ndarray:
        df["LogAmount"] = np.log(df.pop("Amount") + eps)
        return df.values
