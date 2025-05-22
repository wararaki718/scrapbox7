import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
