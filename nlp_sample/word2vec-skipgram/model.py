import torch
import torch.nn as nn


class SimpleCBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self._in_layer1 = nn.Linear(vocab_size, embedding_dim)
        self._in_layer2 = nn.Linear(vocab_size, embedding_dim)
        self._out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h0 = self._in_layer1(x1)
        h1 = self._in_layer2(x2)
    
        h = (h0 + h1) * 0.5
        embeddings = self._out_layer(h)
        return embeddings
