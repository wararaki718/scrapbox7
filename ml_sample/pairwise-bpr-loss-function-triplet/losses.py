from typing import Callable

import torch
import torch.nn as nn


def calc_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.cosine_similarity(x, y)


class PairwiseBPRLoss(nn.Module):
    def __init__(
        self,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        
        if distance_function is None:
            distance_function = calc_similarity
        self._distance_function = distance_function
    
    def __call__(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_positive = self._distance_function(anchor, positive)
        d_negative = self._distance_function(anchor, negative)

        d_diff = (d_positive - d_negative).squeeze()

        return self._bce_loss(d_diff, torch.ones(d_positive.shape[0]))
