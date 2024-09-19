import torch


def get_dummy_image() -> torch.Tensor:
    image = torch.ones((1, 3, 224, 224))
    return image
