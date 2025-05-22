import torch
import torch.nn as nn
from torch.utils.data import  DataLoader

from model.dataset import MultiModalDataset


class Trainer:
    def _train_step(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Adam,
        train_loader: DataLoader,
    ) -> float:
        train_loss = 0.0
        for (
            X_query_context,
            X_query_image,
            X_query_text,
            X_query_action,
            X_document_context,
            X_document_image,
            X_document_text,
            X_document_action,
            y
        ) in train_loader:
            X_query, X_document = model(
                X_query_context,
                X_query_image,
                X_query_text,
                X_query_action,
                X_document_context,
                X_document_image,
                X_document_text,
                X_document_action,
            )
            loss: torch.Tensor = criterion(X_query, X_document, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss

    def train(
        self,
        model: nn.Module,
        dataset: MultiModalDataset,
        n_epochs: int=2,
        batch_size: int=128,
    ) -> None:
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CosineEmbeddingLoss(margin=0.1)

        for epoch in range(1, n_epochs+1):
            loss = self._train_step(model, criterion, optimizer, train_loader)
            print(f"epoch {epoch}: {loss}", flush=True)
