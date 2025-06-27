import numpy as np

from datasets import Dataset
from sklearn.metrics import classification_report


def evaluate(y_true: list[np.int64], y_pred: Dataset) -> str:
    performance: str = classification_report(
        y_true,
        y_pred,
        target_names=["negative review", "positive review"],
    )
    return performance
