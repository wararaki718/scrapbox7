from typing import Any

from sklearn.linear_model import SGDRegressor


def estimator_fn(estimator_params: dict[str, Any] = None):
    if estimator_params is None:
        estimator_params = {}

    return SGDRegressor(random_state=42, **estimator_params)
