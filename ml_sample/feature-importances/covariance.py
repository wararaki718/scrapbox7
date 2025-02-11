import numpy as np


# Correlation with label
def show_covariance(feature_names: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    print("# show covariance")
    for feature_name, x_i in zip(feature_names, X.T):
        covariance = np.cov(x_i, y)
        print(f"{feature_name:<8} {abs(covariance[0][1]):.3f}")
    print()
