import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_bar = y_true.mean() * np.ones_like(y_true)

    bs_ref = brier_score_loss(y_true, y_bar)
    bs = brier_score_loss(y_true, y_pred)

    return 1.0 - bs / bs_ref


def _calib_curve_probs(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=10, method: str="uniform") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = pd.DataFrame({
        "label": y_true,
        "score": y_prob,
    })
    predictions.sort_values("score", inplace=True)

    bins = np.linspace(0, 1, n_bins + 1)
    if method == "quantile":
        quantiles = [predictions["score"].quantile(bin_) for bin_ in bins]
        predictions["bins"] = pd.cut(predictions["score"], bins=quantiles)
    else:
        predictions["bins"] = pd.cut(predictions["score"], bins=bins)

    calibrations = predictions.groupby("bins", observed=False).mean().reset_index(drop=True)
    calibrations.columns = ["label", "score"]

    probabilities = predictions.groupby("bins", observed=False).apply(lambda x: len(x)).values
    probabilities = probabilities / len(y_true)

    return calibrations.score, calibrations.label, probabilities


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=10, method: str="uniform") -> float:
    x, y, p = _calib_curve_probs(y_true, y_prob, n_bins, method)
    return (p * np.abs(x - y)).sum()


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=10, method: str="uniform") -> float:
    x, y, _ = _calib_curve_probs(y_true, y_prob, n_bins, method)
    return np.abs(x - y).max()


# binary classifier only
def stratified_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    positive_score = ((y_prob[y_true == 1] - 1)**2).mean()
    negative_score = (y_prob[y_true == 0]**2).mean()
    return positive_score, negative_score
