import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import try_gpu


def f1_score_macro(model: nn.Module, loader: DataLoader) -> float:
    model.eval()

    labels: torch.Tensor = loader.dataset.get_labels()
    tps = np.zeros(len(labels))
    fps = np.zeros(len(labels))
    fns = np.zeros(len(labels))
    with torch.no_grad():
        for X, y in loader:
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)

            output = model(X)
            _, y_preds = torch.max(output, 1)
            for i, label in enumerate(labels):
                tps[i] += ((y == label) & (y_preds == label)).sum().item()
                fps[i] += ((y != label) & (y_preds == label)).sum().item()
                fns[i] += ((y == label) & (y_preds != label)).sum().item()

    # calc score
    scores = []
    for tp, fp, fn in zip(tps, fps, fns):
        if tp == 0:
            score = 0
        else:
            score = 2*tp / (2*tp + fp + fn)
        scores.append(score)
    return np.mean(scores)


def f1_score_micro(model: nn.Module, loader: DataLoader) -> float:
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    labels: torch.Tensor = loader.dataset.get_labels()
    with torch.no_grad():
        for X, y in loader:
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)

            output = model(X)
            _, y_preds = torch.max(output, 1)
            for label in labels:
                tp += ((y == label) & (y_preds == label)).sum().item()
                fp += ((y != label) & (y_preds == label)).sum().item()
                fn += ((y == label) & (y_preds != label)).sum().item()

    # calc score
    score = 2*tp / (2*tp + fp + fn)
    return score
