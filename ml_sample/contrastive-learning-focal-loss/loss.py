import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, scale: float=1.0, reduction: str="mean") -> None:
        super().__init__()
        self._scale = scale
        self._reduction = reduction

    def forward(self, query: torch.Tensor, document: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        similarities = self._scale * nn.functional.cosine_similarity(query, document)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            similarities,
            y,
            reduction=self._reduction,
        )
        return loss
