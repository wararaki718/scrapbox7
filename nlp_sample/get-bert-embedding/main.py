import torch
from transformers import BertModel, BertTokenizer


def try_gpu(x: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    return x


def main() -> None:
    model_name = "bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    model: BertModel = BertModel.from_pretrained(model_name)
    model = try_gpu(model)

    text = "Here is some text to encode"
    input_ids: torch.Tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    input_ids = try_gpu(input_ids)

    with torch.no_grad():
        last_hidden_states: torch.Tensor = model(input_ids)[0]

    print(last_hidden_states.shape)

    print("DONE")


if __name__ == "__main__":
    main()
