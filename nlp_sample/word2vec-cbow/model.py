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


def collate_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    context_list, target_list = [], []
    max_len = 0
    for context, target in batch:
        max_len = max(max_len, len(context))

    for context, target in batch:
        padded_context = torch.cat([context, torch.zeros(max_len - len(context), dtype=torch.long)])
        context_list.append(padded_context)
        target_list.append(target)

    return torch.stack(context_list), torch.stack(target_list)
