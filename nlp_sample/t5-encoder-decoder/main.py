import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import try_gpu


def main() -> None:
    model_name = "google-t5/t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = try_gpu(model)

    input_ids: torch.Tensor = tokenizer(
        "translate English to German: The house is wonderful.",
        return_tensors="pt",
    ).input_ids
    print(input_ids)
    print()

    input_ids = try_gpu(input_ids)
    outputs: torch.Tensor = model.generate(input_ids).cpu()
    print(outputs)
    print()

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
