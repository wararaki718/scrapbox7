from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        query_encoder: nn.Module,
        document_encoder: nn.Module,
        criterion: nn.TripletMarginWithDistanceLoss,
        query_optimizer: Optimizer,
        document_optimizer: Optimizer,
    ) -> None:
        self._query_encoder = query_encoder
        self._document_encoder = document_encoder
        self._criterion = criterion
        self._query_optimizer = query_optimizer
        self._document_optimizer = document_optimizer
        self._training_loss = []

    def train(self, data_loader: DataLoader, epoch: int) -> None:
        self._query_encoder.train()
        self._document_encoder.train()

        running_loss = 0
        for _, (x_q, x_pos, x_neg) in enumerate(data_loader):
            anchor = self._query_encoder(x_q)
            positive = self._document_encoder(x_pos)
            negative = self._document_encoder(x_neg)

            loss: torch.Tensor = self._criterion(anchor, positive, negative)

            self._query_optimizer.zero_grad()
            self._document_optimizer.zero_grad()

            loss.backward()

            self._query_optimizer.zero_grad()
            self._document_optimizer.zero_grad()

            running_loss += loss.item()

        print(f"epoch {epoch}: loss={running_loss/len(data_loader)}")
        self._training_loss.append(running_loss / len(data_loader))

    def save_model(self, query_model_path: Path, document_model_path: Path) -> None:
        torch.save(self._query_encoder, query_model_path)
        torch.save(self._document_encoder, document_model_path)
