import gc

import torch
import torchvision

from measure import measure
from utils import try_gpu, generate_dummy_images


def main() -> None:
    model = torchvision.models.resnet50()
    print("model loaded!")

    images = generate_dummy_images(n_data=512)
    print("data loaded!")

    result = measure(model, images)
    print(f"cpu: {result} sec")

    if torch.cuda.is_available():
        model = try_gpu(model)
        images = try_gpu(images)
        result = measure(model, images)
        print(f"gpu: {result} sec")

        with torch.autocast(device_type="cuda"):
            result = measure(model, images)
        print(f"gpu + autocast: {result}")
    
    if torch.cuda.is_available():
        model = torchvision.models.resnet50()
        model = try_gpu(model)
        model = torch.compile(model)
        images = try_gpu(images)

        result = measure(model, images)
        print(f"compiled gpu: {result} sec")

    print("DONE")


if __name__ == "__main__":
    main()
