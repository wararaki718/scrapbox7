import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self._embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._embeddings.weight.data.uniform_(-1, 1)

        self._linear = nn.Linear(embedding_dim, vocab_size)
        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        embedded_contexts = self._embeddings(word_indices)
        sum_embedded_contexts = torch.sum(embedded_contexts, dim=1)

        output = self._linear(sum_embedded_contexts)
        log_probs: torch.Tensor = self._log_softmax(output)
        return log_probs

    def get_embeddings(self) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            embeddings: torch.Tensor = self._embeddings.weight.cpu().detach()
        return embeddings
