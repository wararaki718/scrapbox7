import torch
import torch.nn as nn

from model import NNModel


def main() -> None:
    anchor = torch.rand(1, 10)
    positive = torch.rand(1, 10)
    negative = torch.rand(1, 10)
    print(f"anchor  : {anchor.shape}")
    print(f"positive: {positive.shape}")
    print(f"negative: {negative.shape}")
    print()

    model = NNModel()
    triplet_loss = nn.TripletMarginLoss()
    custom_triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=nn.functional.cosine_similarity,
        margin=0.1,
    )

    # output embeddings
    x_anchor = model(anchor)
    print(x_anchor)
    print()

    x_positive = model(positive)
    print(x_positive)
    print()

    x_negative = model(negative)
    print(x_negative)
    print()

    loss = triplet_loss(x_anchor, x_positive, x_negative)
    print(f"triplet_loss: {loss}")
    loss = custom_triplet_loss(x_anchor, x_positive, x_negative)
    print(f"custom_loss : {loss}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
