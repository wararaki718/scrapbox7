from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from ablation import show_ablation
from coef import show_coef
from covariance import show_covariance
from permutation import show_permutation
from shap_ import show_shap
from tree import show_tree


def main() -> None:
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0
    )
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print()

    model = Ridge(alpha=1e-2)
    model.fit(X_train, y_train)

    # show feature importances
    show_coef(diabetes.feature_names, model)
    show_covariance(diabetes.feature_names, X, y)
    show_ablation(diabetes.feature_names, X, y)
    show_permutation(diabetes.feature_names, model, X_test, y_test)
    show_tree(diabetes.feature_names, X_train, y_train)
    show_shap(diabetes.feature_names, X, y)

    print("DONE")


if __name__ == "__main__":
    main()
