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
