import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import AGNewsDataset
from evaluate import Evaluator
from loader import NewsLoader
from model import NNModel, LightSENNModel
from trainer import Trainer
from utils import try_gpu
from vectorizer import BERTVectorizer


def main() -> None:
    # data loader
    sentences, labels = NewsLoader.load(target="train")
    print(f"n_data: {len(sentences)}")
    print(f"labels: {np.unique(labels)}")

    # vectorize
    model_name = "bert-base-uncased"
    vectorizer = BERTVectorizer(model_name=model_name)
    print("vectorizer loaded.")

    chunksize = 512
    embeddings = []
    for i in range(0, len(sentences), chunksize):
        embedding = vectorizer.transform(sentences[i: i+chunksize])
        embeddings.append(embedding)
    X = torch.cat(embeddings)
    print(X.shape)

    # dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
    )
    train_dataset = AGNewsDataset(X_train, torch.Tensor(y_train).long())
    valid_dataset = AGNewsDataset(X_valid, torch.Tensor(y_valid).long())
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")

    # loader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    # nn model
    print("NNModel train:")
    model = NNModel(
        n_input=X.shape[2],
        n_output=len(set(labels)),
    )
    model = try_gpu(model)
    trainer = Trainer(n_epochs=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = trainer.train(model, optimizer, train_loader, valid_loader)
    print("model trained!")
    print()

    # light se model
    print("LightSE train:")
    light_se_model = LightSENNModel(
        n_input=X.shape[2],
        n_output=len(set(labels))
    )
    light_se_model = try_gpu(light_se_model)
    trainer = Trainer(n_epochs=100)
    optimizer = torch.optim.Adam(light_se_model.parameters(), lr=1e-3)
    _ = trainer.train(light_se_model, optimizer, train_loader, valid_loader)
    print("light se model trained!")
    print()

    # evaluate
    print("test:")
    sentences, labels = NewsLoader.load(target="test")
    embeddings = []
    for i in range(0, len(sentences), chunksize):
        embedding = vectorizer.transform(sentences[i: i+chunksize])
        embeddings.append(embedding)
    X_test = torch.cat(embeddings)
    print(X.shape)

    test_dataset = AGNewsDataset(X_test, torch.Tensor(labels).long())
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    evaluator = Evaluator()
    result = evaluator.evaluate(model, test_dataloader)
    se_result = evaluator.evaluate(light_se_model, test_dataloader)
    print(f"nn model: {result}")
    print(f"light se: {se_result}")

    print("DONE")


if __name__ == "__main__":
    main()
