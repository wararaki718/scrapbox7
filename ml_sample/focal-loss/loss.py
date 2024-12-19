import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float=0.25, gamma: float=2.0, reduction: str="none") -> None:
        super().__init__()

        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction="none")
        p_t = p * y + (1 - p) * (1 - y)

        loss = ce_loss * ((1 - p_t)**self._gamma)

        if self._alpha >= 0.0:
            alpha_t = self._alpha * y + (1 - self._alpha) * (1 - y)
            loss = alpha_t * loss

        if self._reduction == "mean":
            return loss.mean()
        
        if self._reduction == "sum":
            return loss.sum()

        return loss
