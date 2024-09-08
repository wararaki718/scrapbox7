import torch


def quantize(x: torch.FloatTensor, w: float=0.1) -> tuple[float, torch.ShortTensor]:
    x = x / w
    return w, x.short()
