from typing import Callable

import torch
import torch.nn as nn


def calc_similarity(x: torch.Tensor, y: torch.Tensor):
    return (1.0 - nn.functional.cosine_similarity(x, y))


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float=1.0,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._ce_loss = nn.CrossEntropyLoss(reduction="mean")

        self._temperature = temperature
        
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

        d = torch.cat([d_positive.view(-1, 1), d_negative.view(-1, 1)], dim=1)
        labels = torch.zeros(d.shape[0], dtype=torch.long)

        return self._ce_loss(d / self._temperature, labels)
