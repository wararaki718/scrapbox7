import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # 埋め込みの重みを一様分布で初期化
        self.embeddings.weight.data.uniform_(-1, 1)

    def forward(self, context_word_idxs):
        # context_word_idxs: (batch_size, num_context_words_in_each_sample)
        # 各コンテキスト単語の埋め込みを取得
        # (batch_size, num_context_words_in_each_sample, embedding_dim)
        embedded_contexts = self.embeddings(context_word_idxs)

        # コンテキスト単語の埋め込みを合計 (または平均)
        # (batch_size, embedding_dim)
        sum_embedded_contexts = torch.sum(embedded_contexts, dim=1)

        # 線形層を通して、各単語のロジットを出力
        # (batch_size, vocab_size)
        output = self.linear(sum_embedded_contexts)

        # 確率分布を得るためにログソフトマックスを適用
        log_probs = nn.functional.log_softmax(output, dim=1)
        return log_probs
