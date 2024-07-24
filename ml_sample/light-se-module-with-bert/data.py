import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vectorizer import BERTVectorizer


def save_embeddings(
    sentences: np.ndarray,
    labels: np.ndarray,
    embedding_dir: Path,
    vectorizer: BERTVectorizer,
    chunksize: int=512,
) -> list:
    index = 0
    for i in tqdm(range(0, len(sentences), chunksize)):
        embedding = vectorizer.transform(sentences[i: i+chunksize])
        chunk_labels = torch.Tensor(labels[i: i+chunksize]).long()

        filepath = embedding_dir / f"embedding_{index}.pt"
        torch.save(embedding, filepath)
        print(f"'{filepath}' is saved.", flush=True)

        filepath = embedding_dir / f"labels_{index}.pt"
        torch.save(chunk_labels, filepath)
        print(f"'{filepath}' is saved.", flush=True)

        index += 1

    return list(embedding_dir.glob("*.pt"))
