import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import IrisDataset
from metrics import accuracy_score, f1_score_macro, precision_score_macro, recall_score_macro
from model import NNModel
from trainer import Trainer
from utils import try_gpu


def main() -> None:
    # load data
    iris = load_iris()
    X = torch.Tensor(iris.data)
    y = torch.Tensor(iris.target).long()

    # setup dataset & dataloader
    print("dataset:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print(f"train: {X_train.shape}, {y_train.shape}")
    print(f"valid: {X_valid.shape}, {y_valid.shape}")
    print(f"test : {X_test.shape}, {y_test.shape}")
    print()

    train_dataset = IrisDataset(X_train, y_train)
    valid_dataset = IrisDataset(X_valid, y_valid)
    test_dataset = IrisDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    model = NNModel(X.shape[1], len(set(y)))
    try_gpu(model)

    # train
    print("train:")
    trainer = Trainer(n_epochs=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer.train(model, optimizer, train_loader, valid_loader)
    print("model trained!")
    print()

    # evaluate
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    score = accuracy_score(model, test_loader)
    print(f"accuracy_score : {score}")

    score = precision_score_macro(model, test_loader)
    print(f"precision_score: {score}")
    
    score = recall_score_macro(model, test_loader)
    print(f"recall_score   : {score}")

    score = f1_score_macro(model, test_loader)
    print(f"f1_score       : {score}")

    print("DONE")


if __name__ == "__main__":
    main()
