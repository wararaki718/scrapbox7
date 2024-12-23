import torch

from loss import FocalLoss


def main() -> None:
    x1 = torch.abs(torch.rand(5, 2))
    x2 = torch.abs(torch.rand(5, 2))
    y = torch.Tensor([1, 0, 1, 0, 1])

    focal_loss = FocalLoss()
    loss = focal_loss(x1, x2, y)
    print(f"focal loss: {loss}")

    print("DONE")


if __name__ == "__main__":
    main()
