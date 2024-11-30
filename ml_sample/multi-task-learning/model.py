import torch
import torch.nn as nn
import lightning as L


class MultiTaskModel(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)
        self._digit = nn.Linear(32, 10)
        self._parity = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._model(x)
        digit = self._digit(x)
        parity = self._parity(x)
        return digit, parity


class LightningModel(L.LightningModule):
    def __init__(self, model: MultiTaskModel) -> None:
        super().__init__()
        self._model = model
        self._criterion = nn.CrossEntropyLoss()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        x = x.view(x.size(0), -1)
        parity_label = label % 2

        digit, parity = self._model(x)
        loss = self._criterion(digit, label) + self._criterion(parity, parity_label)
        return loss
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, label = batch
        parity_label = label % 2
        x = x.view(x.size(0), -1)

        digit, parity = self._model(x)
        digit_loss = self._criterion(digit, label)
        parity_loss = self._criterion(parity, parity_label)
        self.log("digit_loss", digit_loss)
        self.log("parity_loss", parity_loss)


    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self._model.parameters(), lr=1e-3)
