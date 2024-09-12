import torch


def try_gpu(x: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def generate_dummy_images(n_data: int) -> torch.Tensor:
    n_pixel = 3
    height = 224
    width = 224

    images = torch.ones((n_data, n_pixel, height, width))
    return images
