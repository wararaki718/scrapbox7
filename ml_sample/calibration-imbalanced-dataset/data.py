import numpy as np
from sklearn.datasets import make_classification


def get_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=20000,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_repeated=1,
        random_state=10,
    )
    return X, y
