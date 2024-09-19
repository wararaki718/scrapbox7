import time

import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
from torch.onnx import export

from evaluate import evaluate
from utils import get_dummy_image


def export_onnx(
    model: torch.nn.Module,
    output_file: str,
    input_shape: tuple[int, int, int, int]=(1, 3, 224, 224),
) -> str:
    model.eval()
    input_tensor = torch.randn(input_shape)
    input_names = ["input"]
    output_names = ["output"]
    export(
        model,
        input_tensor,
        output_file,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return output_file


def main() -> None:
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    #output_path = "model/resnet50_dense.onnx"
    #_ = export_onnx(model, output_path)
    print(f"saved.")

    image = get_dummy_image()

    print("base:")
    evaluate(model, image)

    # pruning
    pruned_params = [
        (module, "weight") for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    prune.global_unstructured(
        pruned_params,
        pruning_method=prune.L1Unstructured,
        amount=0.9,
    )

    print("prune:")
    evaluate(model, image)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")
    
    print("prune remove:")
    evaluate(model, image)

    print("DONE")


if __name__ == "__main__":
    main()
