from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import save_embeddings
from dataset import AGNewsDataset
from evaluate import Evaluator
from loader import NewsLoader
from model import NNModel, LightSENNModel
from trainer import Trainer
from utils import try_gpu
from vectorizer import BERTVectorizer


def main() -> None:
    # data loader
    sentences, labels = NewsLoader.load(target="train", is_shuffle=True)
    print(f"n_data: {len(sentences)}")
    print(f"labels: {np.unique(labels)}")

    train_sentences, valid_sentences, train_labels, valid_labels = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # vectorize
    model_name = "bert-base-uncased"
    vectorizer = BERTVectorizer(model_name=model_name)
    print("vectorizer loaded.")

    embedding_dir = Path("./data")
    train_embedding_dir = embedding_dir / "train"
    #output_files = save_embeddings(train_sentences, train_labels, train_embedding_dir, vectorizer)
    #print(output_files)

    valid_embedding_dir = embedding_dir / "valid"
    #output_files = save_embeddings(valid_sentences, valid_labels, valid_embedding_dir, vectorizer)
    #print(output_files)

    train_dataset = AGNewsDataset(train_embedding_dir, is_shuffle=True)
    valid_dataset = AGNewsDataset(valid_embedding_dir)
    print("dataset created!")

    # loader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    # nn model
    print("NNModel train:")
    model = NNModel(
        n_input=768,
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
        n_input=768,
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
    test_embedding_dir = embedding_dir / "test"
    output_files = save_embeddings(sentences, labels, test_embedding_dir, vectorizer)
    print(output_files)

    test_dataset = AGNewsDataset(test_embedding_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    evaluator = Evaluator()
    result = evaluator.evaluate(model, test_dataloader)
    se_result = evaluator.evaluate(light_se_model, test_dataloader)
    print(f"nn model: {result}")
    print(f"light se: {se_result}")

    print("DONE")


if __name__ == "__main__":
    main()
