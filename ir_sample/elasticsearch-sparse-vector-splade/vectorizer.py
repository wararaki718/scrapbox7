import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeVectorizer:
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil"
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._index2token = {v: k for k, v in self._tokenizer.get_vocab().items()}

    def transform(self, texts: list[str]) -> list[dict[str, float]]:
        tokens: dict = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self._model(**tokens)
        embeddings = torch.log(1 + torch.relu(outputs.logits)) * tokens.attention_mask.unsqueeze(-1)
        vectors, _ = torch.max(embeddings, dim=1)

        rows, cols = vectors.nonzero(as_tuple=True)
        weights = vectors[rows, cols]

        results: list[dict, float] = []
        for i, _ in enumerate(texts):
            result: dict[str, float] = {
                self._index2token[col.item()]: weight.item()
                for col, weight in zip(cols[rows == i], weights[rows == i])
            }
            results.append(result)

        return results
