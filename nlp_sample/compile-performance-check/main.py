import torch
import openvino.torch
from transformers import BertModel, BertTokenizer

from evaluate import evaluate
from utils import try_gpu, get_texts

def main() -> None:
    model_name = "bert-base-uncased"
    texts = get_texts()

    print("base:")
    base_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    base_model: BertModel = BertModel.from_pretrained(model_name)
    base_model = try_gpu(base_model)
    evaluate(base_model, base_tokenizer, texts)

    print("compile:")
    tokenizer = torch.compile(base_tokenizer)
    model = torch.compile(base_model)
    evaluate(model, tokenizer, texts)
    del model, tokenizer

    print("compile (openvino):")
    tokenizer = torch.compile(base_tokenizer, backend="openvino")
    model = torch.compile(base_model, backend="openvino")
    evaluate(model, tokenizer, texts)
    del model, tokenizer

    print("DONE")


if __name__ == "__main__":
    main()
