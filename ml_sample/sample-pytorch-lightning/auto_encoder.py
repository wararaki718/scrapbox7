import lightning
import torch
import torch.nn as nn


class LightningAutoEncoder(lightning.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(LightningAutoEncoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int) -> torch.Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self._encoder(x)
        x_hat = self._decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_encoder(self) -> nn.Module:
        return self._encoder
