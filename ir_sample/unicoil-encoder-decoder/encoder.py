from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


class DenseContextEncoder:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self._encoder = DPRContextEncoder.from_pretrained(model_name)

    def encode(self, contexts: List[str]) -> torch.Tensor:
        inputs = self._tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
        embeddings = self._encoder(**inputs).pooler_output
        return embeddings


class DenseQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self._encoder = DPRQuestionEncoder.from_pretrained(model_name)
    
    def encode(self, query: str) -> torch.Tensor:
        inputs = self._tokenizer(query, return_tensors="pt")["input_ids"]
        embeddings = self._encoder(inputs).pooler_output
        return embeddings


class SparseContextEncoder:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._encoder = AutoModel.from_pretrained(model_name)

    def encode(self, contexts: List[str]) -> torch.Tensor:
        inputs = self._tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")
        embeddings = self._encoder(**inputs).last_hidden_state[:, 0, :]
        return embeddings


class SparseQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._encoder = AutoModel.from_pretrained(model_name)

    def encode(self, query: str) -> torch.Tensor:
        inputs = self._tokenizer(query, return_tensors="pt")
        embeddings = self._encoder(**inputs).last_hidden_state[:, 0, :]
        return embeddings
