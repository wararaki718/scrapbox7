import torch

from loss import ContrastiveLoss, TripletLoss, InfoNCELoss


def main() -> None:
    x1 = torch.abs(torch.rand(5, 2))
    x2 = torch.abs(torch.rand(5, 2))
    anchor = torch.abs(torch.rand(5, 2))
    y = torch.Tensor([1, 0, 1, 0, 1]).view(-1, 1)

    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss = contrastive_loss(x1, x2, y)
    print(f"contrastive loss: {loss}")

    triplet_loss = TripletLoss()
    loss = triplet_loss(anchor, x1, x2)
    print(f"triplet loss: {loss}")

    info_nce_loss = InfoNCELoss()
    loss = info_nce_loss(anchor, x1, x2)
    print(f"info_nce_loss: {loss}")

    print("DONE")


if __name__ == "__main__":
    main()
