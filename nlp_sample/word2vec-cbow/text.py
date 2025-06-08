from collections import Counter


def preprocess_text(text: str, threshold: int=1) -> tuple[list[str], dict[str, int], dict[int, str]]:
    text = text.lower()
    words = text.split()

    # ボキャブラリーの作成
    word_counts = Counter(words)
    keep_words = [key for key, value in word_counts.items() if value >= threshold]
    vocabularies = sorted(keep_words, key=lambda x: word_counts[x], reverse=True)

    word2index = {word: i for i, word in enumerate(vocabularies)}
    index2word = {i: word for i, word in enumerate(vocabularies)}

    print(f"Vocab Size: {len(word2index)}")
    print(f"Word to Index: {word2index}")

    return words, word2index, index2word
