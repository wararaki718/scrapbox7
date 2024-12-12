import torch
import torch.nn as nn


def main() -> None:
    n_dim = 5
    X = torch.rand(3, n_dim)
    print(X.shape)
    print(X)
    print()

    model = nn.BatchNorm1d(n_dim)

    y = model(X)
    print(y.shape)
    print(y)
    print("DONE")


if __name__ == "__main__":
    main()
