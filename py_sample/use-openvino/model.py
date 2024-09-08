import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int =8) -> None:
        super().__init__()
        layers = [
            nn.Linear(n_input, n_hidden),
            nn.Linear(n_hidden, n_output),
        ]
        self._layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)
