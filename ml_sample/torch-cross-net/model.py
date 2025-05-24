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


class CustomModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.cross_net = CrossNet(n_input, n_layers)
        self.nn_model = NNModel(n_input, n_output, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_crossed = self.cross_net(x)
        return self.nn_model(x_crossed)
    