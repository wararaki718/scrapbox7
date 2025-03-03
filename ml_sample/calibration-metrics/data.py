import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def get_data(
    n_samples: int=50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)
    y[:n_samples//2] = 0
    y[n_samples//2:] = 1

    sample_weights = np.random.RandomState(42).rand(y.shape[0])

    # train test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.9, random_state=42,
    )
    return X_train, X_test, y_train, y_test, w_train, w_test
