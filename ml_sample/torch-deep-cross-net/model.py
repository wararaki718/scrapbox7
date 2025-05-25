import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        layers = [
            nn.Linear(n_input, n_output),
            nn.Sigmoid(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class CrossNet(nn.Module):
    def __init__(self, n_input: int, n_layers: int=3):
        super().__init__()

        layers = []
        biases = []
        norms = []
        for _ in range(n_layers):
            layers.append(nn.Parameter(torch.randn(n_input, n_input)))
            biases.append(nn.Parameter(torch.randn(n_input, 1)))
            norms.append(nn.LayerNorm(n_input))
        self._layers = nn.ParameterList(layers)
        self._biases = nn.ParameterList(biases)
        self._norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = x
        x_i = x
        for i in range(len(self._layers)):
            x_w: torch.Tensor = torch.matmul(self._layers[i], x_i.T)+ self._biases[i]
            x_i: torch.Tensor = self._norms[i](x_0 * x_w.T + x_i)
        return x_i


class DeepNet(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(n_input, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())
            n_input = n_hidden
        layers.append(nn.Linear(n_hidden, n_output))
        layers.append(nn.BatchNorm1d(n_output))
        layers.append(nn.ReLU())
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class DeepCrossNet(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.cross_net = CrossNet(n_input, n_layers)
        self.deep_net = DeepNet(n_input, n_hidden, n_hidden, n_layers)
        self.nn_model = CustomModel(n_input + n_hidden, n_output, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cross = self.cross_net(x)
        x_deep = self.deep_net(x)
        x_stack = torch.cat((x_cross, x_deep), dim=1)
        return self.nn_model(x_stack)
