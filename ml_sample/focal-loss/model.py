import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_output),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
