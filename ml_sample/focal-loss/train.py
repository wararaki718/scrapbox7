import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def _train_step(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Adam,
        train_loader: DataLoader,
    ) -> float:
        train_loss = 0.0
        for X, y in train_loader:
            y_pred = model(X)
            loss: torch.Tensor = criterion(y, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss

    def train(
        self,
        model: nn.Module,
        train_dataset: TensorDataset,
        criterion: nn.Module,
        n_epochs: int=2,
        batch_size: int=128,
    ) -> None:
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, n_epochs+1):
            loss = self._train_step(model, criterion, optimizer, train_loader)
            print(f"epoch {epoch}: {loss}")
