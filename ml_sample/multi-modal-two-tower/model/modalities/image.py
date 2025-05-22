import torch
import torch.nn as nn


class ImageModality(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
