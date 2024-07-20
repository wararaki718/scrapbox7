import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

from dataset import AGNewsDataset
from evaluate import Evaluator
from model import NNModel, LightSENNModel
from trainer import Trainer
from utils import try_gpu


def main() -> None:
    train, test = AG_NEWS()

    labels = []
    sentences = []
    for label, sentence in train:
        labels.append(label)
        sentences.append(sentence)
    labels = np.array(labels) - 1 # scaling 
    print(np.unique(labels))

    sentences = np.array(sentences)
    print(f"n_data: {len(sentences)}")
    
    # vectorized
    vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.01)
    X = vectorizer.fit_transform(sentences)
    print(f"shape: {X.shape}")

    X_train, X_valid, y_train, y_valid = train_test_split(X, labels, test_size=0.2, random_state=42)
    print(f"vocabs: {len(vectorizer.get_feature_names_out())}")
    print()

    # reshape
    X_train: np.ndarray = X_train.toarray().reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_valid: np.ndarray = X_valid.toarray().reshape((X_valid.shape[0], 1, X_valid.shape[1]))

    # create loader
    train_dataset = AGNewsDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
    valid_dataset = AGNewsDataset(torch.Tensor(X_valid), torch.Tensor(y_valid).long())

    # nn model
    print("NNModel train:")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    model = NNModel(
        n_input=X.shape[1],
        n_output=len(set(labels)),
    )
    model = try_gpu(model)
    trainer = Trainer(n_epochs=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = trainer.train(model, optimizer, train_loader, valid_loader)
    print("model trained!")
    print()

    print("LightSE train:")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    light_se_model = LightSENNModel(
        n_input=X.shape[1],
        n_output=len(set(labels)),
    )
    light_se_model = try_gpu(light_se_model)
    trainer = Trainer(n_epochs=100)
    optimizer = torch.optim.Adam(light_se_model.parameters(), lr=1e-3)
    _ = trainer.train(light_se_model, optimizer, train_loader, valid_loader)
    print("model trained!")
    print()

    # test
    print("test:")
    labels = []
    sentences = []
    for label, sentence in test:
        labels.append(label)
        sentences.append(sentence)
    X_test = vectorizer.transform(sentences)
    X_test: np.ndarray = X_test.toarray().reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_test = np.array(labels) - 1 # scaling

    # evaluate
    test_dataset = AGNewsDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())

    evaluator = Evaluator()
    result = evaluator.evaluate(model, DataLoader(test_dataset, batch_size=256, shuffle=False))
    print(f"nn model: {result}")

    result = evaluator.evaluate(light_se_model, DataLoader(test_dataset, batch_size=256, shuffle=False))
    print(f"light se: {result}")

    print("DONE")


if __name__ == "__main__":
    main()
