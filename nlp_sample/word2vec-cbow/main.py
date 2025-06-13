from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from evaluate import Evaluator
from model import CBOWModel, collate_batch
from text import TextProcessor, VocabularyGenerator
from trainer import Trainer
from utils import get_wikipedia, show


def main() -> None:
    # dataset
    dataset = get_wikipedia()
    print(f"Original Text: {len(dataset)} documents")

    # preprocess
    preprocessor = TextProcessor()
    documents = preprocessor.transform(dataset["text"])
    print(f"Processed Documents: {len(documents)} documents")

    generator = VocabularyGenerator()
    vocabularies = generator.generate(documents, threshold=1)
    print(f"Vocabularies: {len(vocabularies)} words")

    dataset = create_dataset(documents, vocabularies, window_size=2)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_batch)
    print(f"Dataset Size: {len(dataset)}")

    # model
    embedding_dim = 256
    model = CBOWModel(len(vocabularies), embedding_dim)
    print(f"Model: CBOW with {len(vocabularies)} words and embedding dimension {embedding_dim}")
    print()

    # train
    trainer = Trainer()
    model: CBOWModel = trainer.train(model, dataloader, n_epochs=10)
    print("Training completed.")
    print()

    # save model
    model_path = Path("./model/cbow_model.pth")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print()

    # evaluate
    evaluator = Evaluator(model.get_embeddings(), word2index=vocabularies)
    
    print(f"Words most similar to 'dog':")
    results = evaluator.find_similar_words("dog")
    show(results)

    print(f"Words most similar to 'fox':")
    results = evaluator.find_similar_words("fox")
    show(results)

    print("DONE")


if __name__ == "__main__":
    main()
