import torch


class Evaluator:
    def __init__(self, embeddings: torch.Tensor, word2index: dict[str, int]) -> None:
        self._embeddings = embeddings
        self._word2index = word2index
        self._index2word = {index: word for word, index in word2index.items()}

    def find_similar_words(
        self,
        word: str,
        top_n: int=5,
    ) -> None:
        word_index = self._word2index.get(word)
        if word_index is None:
            print(f"'{word}' not found in vocabulary.", flush=True)
            return

        word_vector = self._embeddings[word_index].reshape(1, -1)
        similarities = torch.nn.functional.cosine_similarity(word_vector, self._embeddings, dim=1)
        similarities[word_index] = -2

        most_similar_indices = torch.topk(similarities, k=top_n).indices.tolist()

        results = [
            (self._index2word[index], similarities[index]) for index in most_similar_indices
        ]
        return results
