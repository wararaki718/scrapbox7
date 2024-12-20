import numpy as np
import torch
from torch.utils.data import TensorDataset


class DatasetCreator:
    def create(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.Tensor(X).float(),
            torch.Tensor(y).float().unsqueeze(1),
        )
