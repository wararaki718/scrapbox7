import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def plot_scores(
    models: list[BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    _, ax = plt.subplots(figsize=(12, 6), ncols=2)
    for model in models:
        model_name = type(model).__name__
        yprobs = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, yprobs)
        auc = roc_auc_score(y_test, yprobs)

        # PR
        precision, recall, _ = precision_recall_curve(y_test, yprobs)
        average_precision = average_precision_score(y_test, yprobs)

        # plot
        ax[0].plot(fpr, tpr, label=f"{model_name} (AUC: {auc:.2f})")
        ax[1].plot(recall, precision, label=f"{model_name} (AP: {average_precision:.2f})")
    
    # layout
    ax[0].legend()
    ax[1].legend()
    ax[0].plot([0, 1], linestyle="--", color="gray")
    ax[0].set_title("ROC curve")
    ax[1].set_title("Precision-Recall curve")

    plt.tight_layout()
    plt.show()

