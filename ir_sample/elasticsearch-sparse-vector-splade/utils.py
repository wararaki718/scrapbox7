from typing import Any


def get_docs() -> list[dict[str, Any]]:
    docs = [
        {
            "text": "I had some terribly delicious carrots.",
            "impact": [
                {"I": 0.55, "had": 0.4, "some": 0.28, "terribly": 0.01, "delicious": 1.2, "carrots": 0.8},
                {"I": 0.54, "had": 0.4, "some": 0.28, "terribly": 2.01, "delicious": 0.02, "carrots": 0.4},
            ],
            "positive": {"I": 0.55, "had": 0.4, "some": 0.28, "terribly": 0.01, "delicious": 1.2, "carrots": 0.8},
            "negative": {"I": 0.54, "had": 0.4, "some": 0.28, "terribly": 2.01, "delicious": 0.02, "carrots": 0.4},
        },
        {
            "text": "You had some books.",
            "impact": {"You": 0.20, "had": 0.1, "some": 0.2, "books": 0.01},
        },
    ]
    return docs


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
