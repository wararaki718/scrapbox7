from typing import Optional

import torch


def roc_curve(y_true: torch.Tensor, y_score: torch.Tensor, pos_label: Optional[int]=None) -> float:
    y_true = (y_true == pos_label).long()

    indices = torch.argsort(y_score, descending=True)
    y_score = y_score[indices]
    y_true = y_true[indices]

    distinct_value_indices = torch.where(torch.diff(y_score))[0]
    threshold_idxs = torch.concat([distinct_value_indices, torch.Tensor([len(y_true) - 1])]).long()

    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]

    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]

    tps = torch.concat([torch.Tensor([0]), tps])
    fps = torch.concat([torch.Tensor([0]), fps])
    thresholds = torch.concat([torch.Tensor([float("inf")]), thresholds])
    
    tps = tps / tps[-1]
    fps = fps / fps[-1]
    return tps, fps, thresholds


def auc_score(tps: torch.Tensor, fps: torch.Tensor) -> float:
    return torch.trapezoid(tps, fps)



if __name__ == "__main__":
    a = torch.Tensor([1, 1, 2, 2])
    b = torch.Tensor([0.1, 0.4, 0.35, 0.8])
    tps, fps, thresholds = roc_curve(a, b, pos_label=2)
    print(tps)
    print(fps)
    print(thresholds)
    print(auc_score(tps, fps))
    print()
