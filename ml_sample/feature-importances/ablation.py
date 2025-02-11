import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def feature_ablation(X: np.ndarray, y: np.ndarray) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0,
    )
    model = Ridge(alpha=1e-2)
    model.fit(X_train, y_train)

    y_preds = model.predict(X_test)
    score = np.sum(np.abs(y_preds - y_test))
    return score


# feature ablation (check loss differences)
def show_ablation(feature_names: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    print("# show ablation")
    total_loss = feature_ablation(X, y)
    for i, feature_name in enumerate(feature_names):
        columns = [j for j in range(len(feature_names)) if i != j]
        loss_ablation = feature_ablation(X[:, columns], y)
        diff = total_loss - loss_ablation
        print(f"{feature_name:<8} {diff:.3f}")
    print()
