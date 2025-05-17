import numpy as np


class BinaryQuantizer:
    def transform(self, X: np.ndarray) -> np.ndarray:
        x_max = np.max(X)
        x_min = np.min(X)
        X = (X - x_min) / (x_max - x_min)
        return X >= 0.5

