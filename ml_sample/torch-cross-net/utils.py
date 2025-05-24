import torch


def get_data(n_data: int, n_input: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n_data, n_input)
    y = torch.randn(n_data, 1)
    return x, y
