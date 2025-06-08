import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_words(
    word: str,
    embeddings: np.ndarray,
    word2index: dict[str, int],
    index2word: dict[int, str],
    top_n: int=5,
) -> None:
    if word not in word2index:
        print(f"'{word}' not in vocabulary.")
        return

    word_index = word2index[word]
    word_vector = embeddings[word_index].reshape(1, -1)

    similarities = cosine_similarity(word_vector, embeddings)[0]
    # 自分自身を除外
    similarities[word_index] = -1 # 最も低い値に設定

    most_similar_indices = np.argsort(similarities)[::-1][:top_n]
    print(f"Words most similar to '{word}':")
    for idx in most_similar_indices:
        print(f"- {index2word[idx]} (Similarity: {similarities[idx]:.4f})")
