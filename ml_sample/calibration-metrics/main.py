import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

from data import get_data
from evaluate import evaluate


def main() -> None:
    print("calibration metrics:")
    X_train, X_test, y_train, y_test, w_train, w_test = get_data()
    print(X_train.shape, y_train.shape, w_train.shape)
    print(X_test.shape, y_test.shape, w_test.shape)
    print()

    # no calibration
    model = GaussianNB()
    model.fit(X_train, y_train)
    score, skill_score, ece_score, mce_score, ps_score, ns_score = evaluate(model, X_test, y_test, w_test)
    print("# No calibration")
    print(f"- brier_score       : {score:.4f}")
    print(f"- brier_skill_score : {skill_score:.4f}")
    print(f"- ece_score         : {ece_score:.4f}")
    print(f"- mce_score         : {mce_score:.4f}")
    print(f"- stratified_brier_score (positive): {ps_score:.4f}")
    print(f"- stratified_brier_score (negative): {ns_score:.4f}")
    print()

    # isotonic calibration
    model = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=2)
    model.fit(X_train, y_train, sample_weight=w_train)
    score, skill_score, ece_score, mce_score, ps_score, ns_score = evaluate(model, X_test, y_test, w_test)
    print("# Isotonic calibration")
    print(f"- brier_score      : {score:.4f}")
    print(f"- brier_skill_score: {skill_score:.4f}")
    print(f"- ece_score        : {ece_score:.4f}")
    print(f"- mce_score        : {mce_score:.4f}")
    print(f"- stratified_brier_score (positive): {ps_score:.4f}")
    print(f"- stratified_brier_score (negative): {ns_score:.4f}")
    print()

    # sigmoid calibration
    model = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=2)
    model.fit(X_train, y_train, sample_weight=w_train)
    score, skill_score, ece_score, mce_score, ps_score, ns_score = evaluate(model, X_test, y_test, w_test)
    print("# Sigmoid calibration")
    print(f"- brier_score      : {score:.4f}")
    print(f"- brier_skill_score: {skill_score:.4f}")
    print(f"- ece_score        : {ece_score:.4f}")
    print(f"- mce_score        : {mce_score:.4f}")
    print(f"- stratified_brier_score (positive): {ps_score:.4f}")
    print(f"- stratified_brier_score (negative): {ns_score:.4f}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
