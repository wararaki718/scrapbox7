import torch
import openvino.torch

from openvino import Core
from model import NNModel


def main() -> None:
    print(f"use openvino: {Core().available_devices}")

    X = torch.rand(10, 5)
    model = NNModel(5, 2)
    model = torch.compile(model, backend="openvino")
    print("compiled")

    model.eval()
    with torch.no_grad():
        y = model(X)
        print(y)

    print("DONE")


if __name__ == "__main__":
    main()
