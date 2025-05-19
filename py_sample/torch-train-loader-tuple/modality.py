import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(
        self,
        X_context: torch.Tensor,
        X_image: torch.Tensor,
        X_text: torch.Tensor,
        X_action: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self._X_context = X_context
        self._X_image = X_image
        self._X_text = X_text
        self._X_action = X_action
        self._y = y
    
    def __len__(self) -> int:
        return len(self._X_context)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._X_context[index],
            self._X_image[index],
            self._X_text[index],
            self._X_action[index],
            self._y[index],
        )
