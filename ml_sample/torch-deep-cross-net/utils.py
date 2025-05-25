import torch


def get_data(n_data: int, n_input: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n_data, n_input)
    y = torch.randint(0, 2, size=(n_data,), dtype=torch.float32)
    return x, y
