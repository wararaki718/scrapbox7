from typing import Union

import torch
import torch.nn as nn


def try_gpu(x: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x
