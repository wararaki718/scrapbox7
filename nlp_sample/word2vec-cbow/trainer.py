import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def train(self, model: nn.Module, data_loader: DataLoader) -> nn.Module:
        model.train()
        total_loss = 0.0

        for batch in data_loader:
            inputs, targets = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)