from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class CustomDataset(IterableDataset):
    def __init__(self, data_dir: Path, is_shuffle: bool=False) -> None:
        self._data_dir = data_dir
        self._is_shuffle = is_shuffle

    def __iter__(self) -> Generator[Tuple[int, str], None, None]:
        filepaths = sorted(self._data_dir.glob("*.csv"))
        if self._is_shuffle:
            np.random.shuffle(filepaths)

        for filepath in filepaths:
            df = pd.read_csv(filepath)

            indices = np.arange(df.shape[0])
            if self._is_shuffle:
                np.random.shuffle(indices)

            for id_, name in zip(df.id[indices], df.name[indices]):
                yield (id_, name)
