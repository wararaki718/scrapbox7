from urllib.request import urlopen
from PIL import Image


def get_image(
    image_path: str = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
) -> None:
    image = Image.open(urlopen(image_path)).convert("RGB")
    return image
