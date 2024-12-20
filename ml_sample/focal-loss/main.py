import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split

from dataset import DatasetCreator
from evaluate import Evaluator
from loss import FocalLoss
from model import NNModel
from preprocessor import Preprocessor
from train import Trainer
from vectorizer import Vectorizer


def main() -> None:
    df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    print(df.shape)

    preprocessor = Preprocessor()
    df = preprocessor.transform(df)
    y = df.pop("Class").values
    print(df.shape, y.shape)

    vectorizer = Vectorizer()
    X = vectorizer.transform(df)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    creator = DatasetCreator()
    train_dataset = creator.create(X_train, y_train)
    test_dataset = creator.create(X_test, y_test)

    trainer = Trainer()
    evaluator = Evaluator()
    
    print("BCELoss:")
    bce_model = NNModel(n_input=X.shape[1], n_output=1)
    bce_loss = nn.BCEWithLogitsLoss()
    trainer.train(bce_model, train_dataset, bce_loss)
    result = evaluator.evaluate(bce_model, test_dataset)
    print(result)
    print()

    print("FocalLoss:")
    focal_loss = FocalLoss(reduction="mean")
    focal_model = NNModel(n_input=X.shape[1], n_output=1)
    trainer.train(focal_model, train_dataset, focal_loss)
    result = evaluator.evaluate(focal_model, test_dataset)
    print(result)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
