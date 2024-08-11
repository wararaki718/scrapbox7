import os
from pathlib import Path

import lightning
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from auto_encoder import LightningAutoEncoder
from model import NNEncoder, NNDecoder


def main() -> None:
    encoder = NNEncoder()
    decoder = NNDecoder()
    autoencoder = LightningAutoEncoder(encoder, decoder)
    print("model defined!", flush=True)

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)
    print("data loaded!", flush=True)

    trainer = lightning.Trainer(limit_train_batches=100, max_epochs=1, accelerator="gpu")
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
    print("model trained!", flush=True)

    filepath = Path("./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt")
    encoder = NNEncoder()
    decoder = NNDecoder()
    autoencoder = LightningAutoEncoder.load_from_checkpoint(filepath, encoder=encoder, decoder=decoder)
    print("parameter loaded!", flush=True)

    encoder = autoencoder.get_encoder()
    encoder.eval()
    print("get encoder!", flush=True)

    fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    embeddings = encoder(fake_image_batch)
    print(f"predictions: {embeddings}")

    print("DONE")


if __name__ == "__main__":
    main()
