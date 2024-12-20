import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


class Evaluator:
    def evaluate(
        self,
        model: nn.Module,
        test_dataset: TensorDataset,
        batch_size: int=128,
    ) -> np.ndarray:
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        y_trues = []
        labels = []
        with torch.no_grad():
            for X, y in test_loader:
                preds: torch.Tensor = model(X)
                labels.append((preds > 0.5).long())
                y_trues.append(y.long())
        metrics = confusion_matrix(torch.cat(y_trues), torch.cat(labels))
        return metrics
