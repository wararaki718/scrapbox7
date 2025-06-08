import torch


def find_similar_words(
    word: str,
    embeddings: torch.Tensor,
    word2index: dict[str, int],
    index2word: dict[int, str],
    top_n: int=5,
) -> None:
    if word not in word2index:
        print(f"'{word}' not in vocabulary.")
        return

    word_index = word2index[word]
    word_vector = embeddings[word_index].reshape(1, -1)

    similarities = torch.nn.functional.cosine_similarity(word_vector, embeddings, dim=1)
    similarities[word_index] = -2

    most_similar_indices = torch.topk(similarities, k=top_n).indices
    print(f"Words most similar to '{word}':")
    for index in most_similar_indices:
        print(f"- {index2word[index.item()]} (Similarity: {similarities[index.item()]:.4f})")
