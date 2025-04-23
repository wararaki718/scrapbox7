import torch
import torch.nn as nn


class PermutationEquivariant(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self._gamma = nn.Linear(n_input, n_output)
        self._lambda = nn.Linear(n_input, n_output, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm, _ = x.max(1, keepdim=True)
        xm = self._lambda(xm)
        x = self._gamma(x)
        x = x - xm
        return x


class DeepSet(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self._phi = nn.Sequential(
            PermutationEquivariant(n_input, n_output),
            nn.ELU(inplace=True),
        )
        self._rho = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_output, n_output),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(n_output, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self._phi(x)
        x = self._rho(x.mean(1))
        return x
