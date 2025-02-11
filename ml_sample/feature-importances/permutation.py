import numpy as np
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch


# check permutation importances
def show_permutation(feature_names: np.ndarray, model: Ridge, X_test: np.ndarray, y_test: np.ndarray) -> None:
    print("# show permutation")
    result: Bunch = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=0,
    )

    indices = result.importances_mean.argsort()[::-1]
    for index in indices:
        feature_name = feature_names[index]
        importance_mean = result.importances_mean[index]
        importance_std = result.importances_std[index]
        print(f" {feature_name:<8} {importance_mean:.3f} +/- {importance_std:.3f}")
    print()
