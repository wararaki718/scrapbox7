import torch


def get_data(n_data: int, n_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n_data, n_dim, n_dim)
    y = torch.randint(0, 2, (1000,))
    return x, y
