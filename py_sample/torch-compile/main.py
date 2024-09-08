import torch

from model import NNModel


def main() -> None:
    X = torch.rand(10, 5)
    model = NNModel(5, 2)
    model = torch.compile(model)
    print("compiled")

    model.eval()
    with torch.no_grad():
        y = model(X)
        print(y)

    print("DONE")


if __name__ == "__main__":
    main()
