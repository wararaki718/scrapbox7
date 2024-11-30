from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(
    download_dir: Path = Path("./data"),
    batch_size: int=64
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(download_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(download_dir, train=False, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
