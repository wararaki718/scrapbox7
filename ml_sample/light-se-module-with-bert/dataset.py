from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class AGNewsDataset(IterableDataset):
    def __init__(self, data_dir: Path, is_shuffle: bool=False) -> None:
        self._data_dir = data_dir
        self._is_shuffle = is_shuffle

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        embedding_paths = np.array(sorted(self._data_dir.glob("embedding_*.pt")))
        label_paths = np.array(sorted(self._data_dir.glob("labels_*.pt")))

        if self._is_shuffle:
            indices = np.arange(embedding_paths.shape[0])
            np.random.shuffle(indices)
            embedding_paths = embedding_paths[indices]
            label_paths = label_paths[indices]

        for embedding_path, label_path in zip(embedding_paths, label_paths):
            embeddings = torch.load(embedding_path)
            labels = torch.load(label_path)

            indices = np.arange(len(embeddings))
            if self._is_shuffle:
                np.random.shuffle(indices)

            for embedding, label in zip(embeddings[indices], labels[indices]):
                yield (embedding, label)
