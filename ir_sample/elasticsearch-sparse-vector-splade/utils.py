def get_texts() -> list[str]:
    return ["hello world", "text embeddings", "this is a pen"]


def show(response: dict) -> None:
    hits: dict = response["hits"]["hits"]
    for i, hit in enumerate(hits, start=1):
        print(f"## rank={i} ##")
        print(f"id={hit['_id']}")
        print(f"score={hit['_score']}")
        source = hit["_source"]
        print(f"text={source['text']}")
        print()
    print()
