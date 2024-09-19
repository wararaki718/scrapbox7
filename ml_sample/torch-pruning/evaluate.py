import time

import torch


def evaluate(model: torch.nn.Module, image: torch.Tensor) -> None:
    model.eval()

    # warmup
    _ = model(image)

    start_tm = time.time()
    with torch.no_grad():
        _ = model(image)
    end_tm = time.time()

    print(f"time: {end_tm - start_tm} sec")
