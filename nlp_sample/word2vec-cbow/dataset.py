import torch
from torch.utils.data import Dataset


class CBOWDataset(Dataset):
    def __init__(self, cbow_pairs: tuple[list[int], int]) -> None:
        self.cbow_pairs = cbow_pairs

    def __len__(self) -> int:
        return len(self.cbow_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_indices, target_index = self.cbow_pairs[index]
        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)


def create_dataset(words: list[str], word2index: dict[str, int], window_size: int=2) -> CBOWDataset:
    cbow_pairs = []
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


def collate_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    context_list, target_list = [], []
    max_len = 0
    for context, target in batch:
        max_len = max(max_len, len(context))

    for context, target in batch:
        padded_context = torch.cat([context, torch.zeros(max_len - len(context), dtype=torch.long)])
        context_list.append(padded_context)
        target_list.append(target)

    return torch.stack(context_list), torch.stack(target_list)
