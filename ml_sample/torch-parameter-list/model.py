import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128):
        super().__init__()
        self._params = nn.ParameterList([
            nn.Parameter(torch.randn(n_input, n_hidden)),
            nn.Parameter(torch.randn(n_hidden, n_output))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for param in self._params:
            x = torch.matmul(x, param)
            x = torch.relu(x)
        return x
