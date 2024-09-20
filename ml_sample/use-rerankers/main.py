from rerankers import Reranker
from rerankers.results import RankedResults


def show(results: RankedResults) -> None:
    for i, result in enumerate(results, start=1):
        print(f"rank={i}:")
        print(f"  {result.document=}")
        print(f"  {result.score=}")
    print()


def main() -> None:
    # data
    query = "I love you"
    documents = ["I hate you", "I really like you"]
    document_ids = [0, 1]

    # cross encoder
    ranker = Reranker(
        'mixedbread-ai/mxbai-rerank-large-v1',
        model_type='cross-encoder',
    )
    print("cross-encoder model loaded!")

    print("[cross-encoder ranking]")
    results = ranker.rank(
        query=query,
        docs=documents,
        doc_ids=document_ids,
    )
    show(results)

    # t5 model
    ranker = Reranker(
        "unicamp-dl/InRanker-base",
        model_type="t5",
    )
    print("t5 model loaded.")

    print("[t5 model ranking]")
    results = ranker.rank(
        query=query,
        docs=documents,
        doc_ids=document_ids,
    )
    show(results)
    print("DONE")


if __name__ == "__main__":
    main()
