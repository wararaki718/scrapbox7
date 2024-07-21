import torch


def try_gpu(x: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    return x
