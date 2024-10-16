from time import sleep

from client import ESClient
from loader import MappingLoader


def show(response: dict) -> None:
    hits: list[dict] = response.get("hits").get("hits")
    for i, hit in enumerate(hits, start=1):
        print(f"{i}-th: id={hit['_id']}, score={hit['_score']}")
    print()


def check_search(client: ESClient, mappings: dict, source: str) -> None:
    print("create index")
    index_name = "custom-index100"
    response = client.create_index(index_name, mappings)
    # print(response)
    print()
    sleep(2.0)

    print("data insert")
    doc1 = {"l2_vector": [1.0, 0.0, 0.0], "cosine_vector": [0.0, 0.0, 1.0]}
    response = client.insert(index_name, 1, doc1)
    # print(response)

    doc2 = {"l2_vector": [1.0, 1.0, 0.0], "cosine_vector": [1.0, 1.0, 0.0]}
    response = client.insert(index_name, 2, doc2)
    # print(response)

    doc3 = {"l2_vector": [0.0, 0.0, 1.0], "cosine_vector": [1.0, 0.0, 1.0]}
    response = client.insert(index_name, 3, doc3)
    # print(response)
    # print()
    sleep(2.0)

    print("search")
    query = {
        "knn": {
            "query_vector": [1.0, 0.0, 0.0],
            "field": "l2_vector",
            "k": 3,
            "num_candidates": 5,
        }
    }
    response = client.search(index_name, query, None)
    show(response)
    sleep(2.0)

    print("rescore:")
    rescore_query = {
        "window_size": 2,
        "query": {
            "rescore_query": {
                "function_score": {
                    "script_score": {
                        "script": {
                            "source": source,
                            "params": {
                                "query_vector": [1.0, 0.0, 0.0],
                            }
                        }
                    }
                }
            }
        }
    }
    response = client.search(index_name, query, rescore_query)
    show(response)
    sleep(2.0)

    print("delete index")
    response = client.delete_index(index_name)
    # print(response)
    print()

def main() -> None:
    host = "http://localhost:9200"
    client = ESClient(hosts=[host])

    filepath = "jsons/mappings.json"
    mappings = MappingLoader.load(filepath)

    print("### cosine_similarity")
    source = "cosineSimilarity(params.query_vector, 'cosine_vector')"
    check_search(client, mappings, source)

    print("### dot_product")
    source = "dotProduct(params.query_vector, 'cosine_vector')"
    check_search(client, mappings, source)

    print("### l1_norm")
    source = "l1norm(params.query_vector, 'cosine_vector')"
    check_search(client, mappings, source)

    print("### l2_norm")
    source = "l2norm(params.query_vector, 'cosine_vector')"
    check_search(client, mappings, source)

    print("DONE")


if __name__ == "__main__":
    main()
