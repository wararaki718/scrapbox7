from datasets import load_dataset


def get_wikipedia() -> dict:
    dataset = load_dataset("range3/wikipedia-ja-20230101")
    train = dataset["train"]
    return train


def get_texts() -> list[str]:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog barks.",
        "I love programming in Python.",
        "Natural language processing is fascinating.",
        "Word embeddings are useful for many NLP tasks.",
    ]
    return texts


def show(results: list[tuple[str, float]]) -> None:
    for word, similarity in results:
        print(f"- Word: {word}, Similarity: {similarity:.4f}")
    print()
