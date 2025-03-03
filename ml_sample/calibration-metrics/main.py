import numpy as np
from sklearn.metrics import brier_score_loss


def main() -> None:
    # labels
    y_true = np.array([0, 1, 1, 0])
    y_true_categorical = np.array(["spam", "ham", "ham", "spam"])

    # probalities
    y_prob = np.array([0.1, 0.9, 0.8, 0.4])

    # regression
    score = brier_score_loss(y_true, y_prob)
    print(f"Brier Score: {score} (regression)")

    # classifier
    score = brier_score_loss(y_true, 1.0 - y_prob, pos_label=0)
    print(f"Brier Score: {score} (classifier)")

    # classifier
    score = brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
    print(f"Brier Score: {score} (classifier categorical)")

    print("DONE")


if __name__ == "__main__":
    main()
