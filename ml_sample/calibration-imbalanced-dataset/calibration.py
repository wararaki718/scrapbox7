import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def calibration_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int=10,
    method: str="quantile",
) -> pd.DataFrame:
    y_probs = model.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame({
        "label": y_test,
        "score": y_probs,
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

    return calibrations


def calibration_curve_probs(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int=10,
    method: str="uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_probs = model.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame({
        "label": y_test,
        "score": y_probs,
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
    probabilities = probabilities / len(y_test)

    return calibrations.score, calibrations.label, probabilities


def plot_calibration_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int=10,
    method: str="uniform",
    ax: plt.Axes | None=None,
    legend: str | None=None,
) -> None:
    labels, scores, probabilities = calibration_curve_probs(
        model, X_test, y_test, n_bins, method
    )
    probabilities = probabilities / probabilities.max() + probabilities.min()

    if ax is None:
        plt.scatter(labels, scores, s=100 * probabilities, marker='o', edgecolors='black')
        plt.plot(labels, scores, label=legend)
        plt.plot([0, 1], linestyle='--', color='gray')
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        if legend is not None:
            plt.legend()
    
    else:
        ax.scatter(labels, scores, s=100 * probabilities, marker='o', edgecolors='black')
        ax.plot(labels, scores, label=legend)
        ax.plot([0, 1], linestyle='--', color='gray')
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        if legend is not None:
            ax.legend()
