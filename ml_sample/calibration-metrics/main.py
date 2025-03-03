import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

from data import get_data
from evaluate import evaluate


def main() -> None:
    print("calibration metrics: brier score (smaller is better)")
    X_train, X_test, y_train, y_test, w_train, w_test = get_data()
    print(X_train.shape, y_train.shape, w_train.shape)
    print(X_test.shape, y_test.shape, w_test.shape)
    print()

    # no calibration
    model = GaussianNB()
    model.fit(X_train, y_train)
    score = evaluate(model, X_test, y_test, w_test)
    print(f"No calibration: {score:.4f}")

    # isotonic calibration
    model = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=2)
    model.fit(X_train, y_train, sample_weight=w_train)
    score = evaluate(model, X_test, y_test, w_test)
    print(f"Isotonic calibration: {score:.4f}")

    # sigmoid calibration
    model = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=2)
    model.fit(X_train, y_train, sample_weight=w_train)
    score = evaluate(model, X_test, y_test, w_test)
    print(f"Sigmoid calibration: {score:.4f}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
