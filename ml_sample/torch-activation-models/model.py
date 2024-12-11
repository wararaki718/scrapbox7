import torch
import torch.nn as nn


class SigmoidModel(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        
        layers = [
            nn.Linear(n_input, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, n_output),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class SoftmaxModel(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        
        layers = [
            nn.Linear(n_input, 16),
            nn.Softmax(),
            nn.Linear(16, 16),
            nn.Softmax(),
            nn.Linear(16, n_output),
            nn.Softmax(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class ReLUModel(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        
        layers = [
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_output),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class SoftplusModel(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        
        layers = [
            nn.Linear(n_input, 16),
            nn.Softplus(),
            nn.Linear(16, 16),
            nn.Softplus(),
            nn.Linear(16, n_output),
            nn.Softplus(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
