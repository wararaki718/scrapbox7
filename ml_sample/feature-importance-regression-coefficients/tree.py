import numpy as np
from sklearn.ensemble import RandomForestRegressor


def show_tree(feature_names: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> None:
    print("# show tree based importances")
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    for feature_name, importance in zip(feature_names, model.feature_importances_):
        print(f"{feature_name:<8} {importance:.3f}")
    print()
