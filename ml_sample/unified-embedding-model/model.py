import torch
import torch.nn as nn
import torch.functional as F


class QueryEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(QueryEncoder, self).__init__()
        layers = [
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class DocumentEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(DocumentEncoder, self).__init__()
        layers = [
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
