import torch

from loss import FocalLoss


def main() -> None:
    focal_loss = FocalLoss(reduction="mean")
    y_pred = torch.rand(10)
    y_true = torch.rand(10)

    loss = focal_loss(y_pred, y_true)
    print(loss)
    print("DONE")


if __name__ == "__main__":
    main()
