import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = [
            nn.Linear(10, 4),
            nn.Linear(4, 8),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
