import gc

import torch
import torchvision

from measure import measure
from utils import generate_dummy_images


def main() -> None:
    model = torchvision.models.resnet50()
    model.eval()
    images = generate_dummy_images(n_data=512)
    print("model & data loaded!")

    result = measure(model, images)
    print(f"cpu: {result} sec")

    # half
    model = model.half()
    images = images.half()
    print("model & data loaded!")

    result = measure(model, images)
    print(f"half: {result} sec")
    print()

    del model, images
    gc.collect()

    # bf16
    model = torchvision.models.resnet50()
    model.eval()
    model = model.bfloat16()
    images = generate_dummy_images(n_data=512)
    images = images.bfloat16()
    print("model & data loaded!")

    result = measure(model, images)
    print(f"cpu: {result} sec")
    del model, images
    gc.collect()

    # tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 =True
    model = torchvision.models.resnet50()
    model.eval()
    images = generate_dummy_images(n_data=512)
    print("model & data loaded!")

    result = measure(model, images)
    print(f"cpu: {result} sec")
    del model, images
    gc.collect()



    print("DONE")


if __name__ == "__main__":
    main()
