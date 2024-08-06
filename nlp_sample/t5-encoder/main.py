import torch
from transformers import T5Tokenizer, T5EncoderModel

from utils import try_gpu


def main() -> None:
    model_name = "google-t5/t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model = try_gpu(model)

    input_ids: torch.Tensor = tokenizer(
        "Studies have been shown that owning a dog is good for you",
        return_tensors="pt",
    ).input_ids
    print(input_ids)
    print()

    input_ids = try_gpu(input_ids)
    outputs: torch.Tensor = model(input_ids)
    print(outputs)
    print()

    result = outputs.last_hidden_state.cpu()
    print(result.shape)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
