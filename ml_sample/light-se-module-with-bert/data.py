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
    chunksize: int=500,
) -> list:
    embeddings = []
    chunk_labels = []
    index = 0
    for i in tqdm(range(0, len(sentences), chunksize)):
        embedding = vectorizer.transform(sentences[i: i+chunksize])
        embeddings.append(embedding)
        chunk_labels.append(torch.Tensor(labels[i: i+chunksize]).long())
    
        if len(embeddings) * chunksize < 2000:
            continue

        filepath = embedding_dir / f"embedding_{index}.pt"
        torch.save(torch.cat(embeddings), filepath)
        print(f"'{filepath}' is saved.", flush=True)

        filepath = embedding_dir / f"labels_{index}.pt"
        torch.save(torch.cat(chunk_labels), filepath)
        print(f"'{filepath}' is saved.", flush=True)

        del embeddings, chunk_labels
        gc.collect()
        embeddings = []
        chunk_labels = []
        index += 1

    if len(embeddings) > 0:
        filepath = embedding_dir / f"embedding_{index}.pt"
        torch.save(torch.cat(embeddings), filepath)
        print(f"'{filepath}' is saved.", flush=True)

        filepath = embedding_dir / f"label_{index}.pt"
        torch.save(torch.cat(chunk_labels), filepath)
        print(f"'{filepath}' is saved.", flush=True)

    return list(embedding_dir.glob("*.pt"))
