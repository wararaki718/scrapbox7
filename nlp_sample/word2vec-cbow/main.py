import numpy as np
from torch.utils.data import DataLoader

from dataset import create_dataset, collate_batch
from evaluate import find_similar_words
from model import CBOWModel
from text import preprocess_text
from trainer import Trainer


def main() -> None:
    # サンプルテキスト
    text = "The quick brown fox jumps over the lazy dog. The dog barks."
    print(f"Original Text: {text}")

    words, word2index, index2word = preprocess_text(text, threshold=1)
    print(f"Word to Index: {word2index}")
    print(f"Index to Word: {index2word}")

    dataset = create_dataset(words, word2index, window_size=2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
    print(f"Dataset Size: {len(dataset)}")

    # model
    embedding_dim = 100
    model = CBOWModel(len(word2index), embedding_dim)

    # train
    trainer = Trainer()
    model = trainer.train(model, dataloader, n_epochs=10)
    print("Training completed.")

    # evaluate
    embeddings: np.ndarray =model.embeddings.weight.data.cpu().numpy()
    find_similar_words("dog", embeddings, word2index=word2index, index2word=index2word)
    find_similar_words("fox", embeddings, word2index=word2index, index2word=index2word)

    print("DONE")


if __name__ == "__main__":
    main()
