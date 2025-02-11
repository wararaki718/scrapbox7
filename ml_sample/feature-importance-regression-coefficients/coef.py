import numpy as np
from sklearn.linear_model import Ridge


# Regression Coefficients
def show_coef(feature_names: np.ndarray, model: Ridge) -> None:
    print("# show coef")
    for feature_name, coef in zip(feature_names, model.coef_):
        print(f"{feature_name:<8} {abs(coef):.3f}")
    print()
