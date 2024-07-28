from typing import Tuple

import torch
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._X = X
        self._y = y

    def __len__(self) -> int:
        return len(self._X)
    
    def get_labels(self) -> torch.Tensor:
        return torch.unique(self._y)
    
    def shape(self) -> Tuple[int, int, int]:
        return self._X.shape
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X[index], self._y[index]
