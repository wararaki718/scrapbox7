import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB

from metrics import brier_skill_score, expected_calibration_error, maximum_calibration_error


def evaluate(
    model: CalibratedClassifierCV | GaussianNB,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_test: np.ndarray,
) -> tuple[float, float]:
    y_prob = model.predict_proba(X_test)[:, 1]

    # samller is better
    score = brier_score_loss(y_test, y_prob, sample_weight=w_test)

    # higher is better
    skill_score = brier_skill_score(y_test, y_prob)

    # smaller is better
    ece_score = expected_calibration_error(y_test, y_prob)

    # smaller is better
    mce_score = maximum_calibration_error(y_test, y_prob)

    return score, skill_score, ece_score, mce_score
