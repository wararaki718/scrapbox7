import torch
from transformers import BertModel, BertTokenizer

from utils import try_gpu


class BERTVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self._model: BertModel = BertModel.from_pretrained(model_name)
        self._model = try_gpu(self._model)
    
    def transform(self, texts: list[str]) -> torch.Tensor:
        output: dict = self._tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = try_gpu(output["input_ids"])

        with torch.no_grad():
            last_hidden_states: torch.Tensor = self._model(input_ids)[0]
        return last_hidden_states.cpu().detach()
    