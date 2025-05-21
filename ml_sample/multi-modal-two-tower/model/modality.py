from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch.utils.data import Dataset

ModalitiesType: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

@dataclass
class Modalities:
    X_context: torch.Tensor
    X_image: torch.Tensor
    X_text: torch.Tensor
    X_action: torch.Tensor

    def get_modalities(self, index: int) -> ModalitiesType:
        return (
            self.X_context[index],
            self.X_image[index],
            self.X_text[index],
            self.X_action[index],
        )


class MultiModalDataset(Dataset):
    def __init__(
        self,
        query_modalities: Modalities,
        document_modalities: Modalities,
        y: torch.Tensor,
    ) -> None:
        self._query_modalities = query_modalities
        self._document_modalities = document_modalities
        self._y = y
    
    def __len__(self) -> int:
        return len(self._y)
    
    def __getitem__(self, index: int) -> tuple[ModalitiesType, ModalitiesType, torch.Tensor]:
        return self._query_modalities.get_modalities(index) + self._document_modalities.get_modalities(index) + (self._y[index],)
