import torch
import torch.nn as nn


class LightSE(nn.Module):
    def __init__(self, in_features: int, out_features: int=32) -> None:
        super(LightSE, self).__init__()
        self._softmax = nn.Softmax(dim=1)
        self._in_features = in_features
        self._out_features = out_features # not used?
        self._linear = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # z = torch.mean(x, dim=-1, out=None)
        # print(z.shape)
        a = self._linear(x)
        a = self._softmax(a)
        # print(a.shape)
        out = x * torch.unsqueeze(a, dim=0)
        # print(out.shape)
        return x + out


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=128) -> None:
        super(NNModel, self).__init__()
        layers = [
            LightSE(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
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
