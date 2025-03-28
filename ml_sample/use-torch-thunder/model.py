import torch
import torch.nn as nn


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
