import torch
import torch.nn as nn


class NNEncoder(nn.Module):
    def __init__(self) -> None:
        super(NNEncoder, self).__init__()
        layers = [
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        ]
        self._model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class NNDecoder(nn.Module):
    def __init__(self) -> None:
        super(NNDecoder, self).__init__()
        layers = [
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
