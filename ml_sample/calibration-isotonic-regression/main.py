from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    print(X.shape)
    print(y.shape)
    print()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print()

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"linear: {score}")

    model = CalibratedClassifierCV(method="isotonic")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"isotonic: {score}")

    print("DONE")


if __name__ == "__main__":
    main()
