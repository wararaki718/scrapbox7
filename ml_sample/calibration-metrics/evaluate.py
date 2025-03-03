import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB


def evaluate(
    model: CalibratedClassifierCV | GaussianNB,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_test: np.ndarray,
) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    score = brier_score_loss(y_test, y_prob, sample_weight=w_test)
    return score
