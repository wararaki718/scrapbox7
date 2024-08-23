from time import sleep

from client import ESClient
from loader import MappingLoader


def main() -> None:
    host = "http://localhost:9200"
    client = ESClient(hosts=[host])

    filepath = "jsons/mappings.json"
    mappings = MappingLoader.load(filepath)

    print("create index")
    index_name = "custom-index"
    response = client.create_index(index_name, mappings)
    print(response)
    print()
    sleep(2.0)

    print("data insert")
    doc1 = {"custom_vector": [1.0, 0.0, 0.0]}
    response = client.insert(index_name, 1, doc1)
    print(response)

    doc2 = {"custom_vector": [0.0, 1.0, 0.0]}
    response = client.insert(index_name, 2, doc2)
    print(response)

    doc3 = {"custom_vector": [0.0, 0.0, 1.0]}
    response = client.insert(index_name, 3, doc3)
    print(response)
    print()
    sleep(2.0)

    print("search")
    query = {
        "knn": {
            "query_vector": [1.0, 0.0, 0.0],
            "field": "custom_vector",
            "k": 3,
            "num_candidates": 5,
        }
    }
    response = client.search(index_name, query)
    print(response)
    print()
    sleep(2.0)

    print("delete index")
    response = client.delete_index(index_name)
    print(response)
    print()

    print("DONE")


if __name__ == "__main__":
    main()