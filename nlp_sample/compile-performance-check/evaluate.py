import time

import torch
from transformers import BertModel, BertTokenizer

from utils import try_gpu


def evaluate(model: BertModel, tokenizer: BertTokenizer, texts: list[str]) -> None:
    start_tm = time.time()
    for text in texts:
        input_ids: torch.Tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        input_ids = try_gpu(input_ids)

        with torch.no_grad():
            _ = model(input_ids)[0]
    end_tm = time.time()
    print(f"encode time: {end_tm - start_tm} sec")

    start_tm = time.time()
    output: dict = tokenizer.batch_encode_plus([text, text], add_special_tokens=True, return_tensors="pt")
    input_ids = try_gpu(output["input_ids"])

    with torch.no_grad():
        _ = model(input_ids)[0]
    end_tm = time.time()
    print(f"batch_encode time: {end_tm - start_tm} sec")
    print()
