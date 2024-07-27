import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import try_gpu


def precision_score_macro(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)

            y_pred = model(X)
            _, y_labels = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (y_labels == y).sum().item()
    return correct / total
