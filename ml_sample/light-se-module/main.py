import numpy as np
import scipy.sparse as sps
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from fastembed import TextEmbedding
from sklearn.model_selection import train_test_split
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

from dataset import AGNewsDataset
from evaluate import Evaluator
from model import NNModel
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
    print()
    
    # vectorized
    # vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.01)
    model_name = "BAAI/bge-small-en-v1.5"
    text_embedder = TextEmbedding(model_name=model_name)
    # X = vectorizer.fit_transform(sentences)
    # X = list(text_embedder.embed(sentences))
    train_sentences, valid_sentences, y_train, y_valid = train_test_split(sentences, labels, test_size=0.2, random_state=42)

    # X_train, X_valid, y_train, y_valid = train_test_split(X, labels, test_size=0.2, random_state=42)
    # print(f"vocabs: {len(vectorizer.get_feature_names_out())}")
    X_train = [embedding for embedding in text_embedder.embed(train_sentences)]
    X_valid = [embedding for embedding in text_embedder.embed(valid_sentences)]
    print()

    # create loader
    train_dataset = AGNewsDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
    valid_dataset = AGNewsDataset(torch.Tensor(X_valid), torch.Tensor(y_valid).long())

    # shuffled model
    print("shuffled dataset:")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    shuffled_model = NNModel(
        n_input=train_dataset.shape[1],
        n_output=len(set(labels)),
    )
    shuffled_model = try_gpu(shuffled_model)
    trainer = Trainer(n_epochs=100)
    optimizer = torch.optim.Adam(shuffled_model.parameters(), lr=1e-3)
    trainer.train(shuffled_model, optimizer, train_loader, valid_loader)
    print("model trained!")
    print()

    # test
    print("test:")
    labels = []
    sentences = []
    for label, sentence in test:
        labels.append(label)
        sentences.append(sentence)
    X_test = [embedding for embedding in text_embedder.embed(sentences)]
    y_test = np.array(labels) - 1 # scaling

    # evaluate
    test_dataset = AGNewsDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())
    evaluator = Evaluator()

    result = evaluator.evaluate(shuffled_model, DataLoader(test_dataset, batch_size=256, shuffle=False))
    print(f"shuffled model: {result}")

    print("DONE")


if __name__ == "__main__":
    main()
