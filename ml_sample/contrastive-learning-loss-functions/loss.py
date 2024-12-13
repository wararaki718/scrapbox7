import torch
import torch.nn as nn


def calc_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return nn.functional.cosine_similarity(x1, x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float=1.0) -> None:
        super().__init__()

        self._distance = calc_distance
        self._margin = margin
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distances = self._distance(x1, x2)
        loss = y * torch.pow(distances, 2.0) + (1 - y) * torch.pow(torch.clamp(self._margin - distances, 0.0), 2.0)
        return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin: float=1.0) -> None:
        super().__init__()

        self._distance = calc_distance
        self._margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_distances = self._distance(anchor, positive)
        negative_distances = self._distance(anchor, negative)

        loss = torch.sum(torch.clamp(positive_distances - negative_distances + self._margin, 0.0))
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._distance = calc_distance

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_distances = self._distance(anchor, positive)
        negative_distances = self._distance(anchor, negative)

        distances = torch.cat([positive_distances, negative_distances])
        labels = torch.cat([torch.ones(len(positive_distances)), torch.zeros(len(negative_distances))])
        loss = nn.functional.cross_entropy(distances, labels)
        return loss
