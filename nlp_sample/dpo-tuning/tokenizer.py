from transformers import AutoTokenizer


def get_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T") -> AutoTokenizer:
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<PAD>"

    return tokenizer
