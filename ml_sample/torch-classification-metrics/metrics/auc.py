import torch


def roc_curve(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    indices = torch.argsort(y_score, descending=True)
    y_score = y_score[indices]
    y_true = y_true[indices]
    distinct_value_indices = torch.where(torch.diff(y_score))[0]
    print(distinct_value_indices)
    threshold_idxs = torch.concat([distinct_value_indices, torch.Tensor([len(y_true) - 1])]).long()
    print(threshold_idxs)

    tps = torch.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return tps, fps, y_score[threshold_idxs]


def auc_score(tps: torch.Tensor, fps: torch.Tensor) -> float:
    return torch.trapezoid(tps, fps)



if __name__ == "__main__":
    a = torch.Tensor([1, 1, 2, 2])
    b = torch.Tensor([0.1, 0.4, 0.35, 0.8])
    tps, fps, _ = roc_curve(a, b)
    print(auc_score(fps, tps))
