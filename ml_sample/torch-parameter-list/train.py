import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self) -> None:
        pass

    def train(self, model: torch.nn.Module, dataset: TensorDataset, n_epoch: int = 10, batch_size: int = 32) -> None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for epoch in range(1, n_epoch+1):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}/{n_epoch}, Loss: {loss.item():.4f}", flush=True)
