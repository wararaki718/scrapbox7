import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import try_gpu


def recall_score_macro(model: nn.Module, loader: DataLoader) -> float:
    model.eval()

    score = 0.0
    total = 0
    labels: torch.Tensor = loader.dataset.get_labels()
    with torch.no_grad():
        for X, y in loader:
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)

            output = model(X)
            _, y_preds = torch.max(output, 1)
            # recall average=macro
            for label in labels:
                tp = ((y == label) & (y_preds == label)).sum()
                fn = ((y == label) & (y_preds != label)).sum()
                if tp > 0:
                    score += (tp / (tp + fn))
            total += 1
    return (score / len(labels)) / total
