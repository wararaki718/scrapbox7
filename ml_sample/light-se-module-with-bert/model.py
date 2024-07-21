import torch
import torch.nn as nn

class LightSE(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(LightSE, self).__init__()
        self._softmax = nn.Softmax(dim=1)
        self._in_features = in_features
        self._linear = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.mean(x, dim=1, out=None)
        a = self._linear(z)
        a = self._softmax(a)
        out = x * torch.unsqueeze(a, dim=1)
        out = torch.flatten(x + out, start_dim=1)
        return out


class LightSENNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=128) -> None:
        super(LightSENNModel, self).__init__()
        layers = [
            LightSE(n_input),
            nn.Linear(n_input, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_output),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=128) -> None:
        super(NNModel, self).__init__()
        layers = [
            nn.Linear(n_input, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_output),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self._model(x)
