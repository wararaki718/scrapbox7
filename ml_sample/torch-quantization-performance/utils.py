import torch


def generate_dummy_images(n_data: int) -> torch.Tensor:
    n_pixel = 3
    height = 224
    width = 224

    images = torch.ones((n_data, n_pixel, height, width))
    return images
