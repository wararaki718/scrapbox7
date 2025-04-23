import torch

from model import DeepSet
from utils import get_data


def main() -> None:
    n_dim = 10
    x, _ = get_data(1000, n_dim)
    print(x.shape) # n_data, x_dim, y_dim

    n_output = 10
    model = DeepSet(n_dim, n_output)
    y = model(x)
    print(y.shape)
    print(y)

    print("DONE")


if __name__ == "__main__":
    main()
