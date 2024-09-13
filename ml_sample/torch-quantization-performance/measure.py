import time

import torch


def measure(model: torch.nn.Module, images: torch.Tensor) -> float:
    torch.cuda.synchronize()
    start_tm = time.time()
    with torch.no_grad():
        _ = model(images)
    
    torch.cuda.synchronize()
    end_tm = time.time()

    return end_tm - start_tm
