import torch
from torch.utils.data import Dataset

from document import Document


class CBOWDataset(Dataset):
    def __init__(self, cbow_pairs: tuple[list[int], int]) -> None:
        self._cbow_pairs = cbow_pairs

    def __len__(self) -> int:
        return len(self._cbow_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_indices, target_index = self._cbow_pairs[index]
        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)


def create_dataset(documents: list[Document], word2index: dict[str, int], window_size: int=2) -> CBOWDataset:
    cbow_pairs = []
    for document in documents:
        words = document.get_words()

        for i, target_word in enumerate(words):
            target_index = word2index[target_word]
            context_indices: list[int] = []
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    context_indices.append(word2index[words[j]])
            
            if context_indices: # コンテキスト単語が存在する場合のみ追加
                cbow_pairs.append((context_indices, target_index))

    dataset = CBOWDataset(cbow_pairs)
    return dataset
