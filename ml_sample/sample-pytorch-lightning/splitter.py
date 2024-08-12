from typing import Tuple

import torch
from torch.utils.data import random_split


def train_test_split(train_set: torch.Tensor, train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    train_set_size = int(len(train_set) * train_ratio)
    valid_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(
        train_set,
        (train_set_size, valid_set_size),
        generator=seed,
    )
    return train_set, valid_set
