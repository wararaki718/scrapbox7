import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(
        self,
        triplet_indices: list[tuple[int, int, int]],
        X_queries: torch.Tensor,
        X_documents: torch.Tensor,
    ) -> None:
        self._triplet_indices = triplet_indices
        self._X_queries = X_queries
        self._X_documents = X_documents

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_index, positive_index, negative_index = self._triplet_indices[index]
        return (
            self._X_queries[query_index],
            self._X_documents[positive_index],
            self._X_documents[negative_index],
        )

    def __len__(self) -> int:
        return len(self._triplet_indices)
