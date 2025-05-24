import torch

from model import CustomModel
from train import Trainer
from utils import get_data


def main() -> None:
    n_data = 1000
    n_input = 10
    n_output = 1

    X, y = get_data(n_data, n_input)
    print(X.shape, y.shape)

    dataset = torch.utils.data.TensorDataset(X, y)
    model = CustomModel(n_input, n_output)
    print("model defined!")

    trainer = Trainer()
    trainer.train(model, dataset, n_epoch=10, batch_size=32)

    print("DONE")


if __name__ == "__main__":
    main()
