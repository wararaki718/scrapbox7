from time import sleep

from client import ESClient
from loader import MappingLoader


def main() -> None:
    host = "http://localhost:9200"
    client = ESClient(hosts=[host])

    filepath = "jsons/mappings.json"
    mappings = MappingLoader.load(filepath)

    print("create index:")
    index_name = "custom-index"
    response = client.create_index(index_name, mappings=mappings)
    print(response)
    print()
    sleep(2.0)

    print("data insert")
    doc1 = {"query": {"match": {"message": "bonsai tree"}}}
    response = client.insert(index_name, 1, doc1)
    print(response)
    print()
    sleep(2.0)

    print("search:")
    query = {
        "query": {
            "percolate": {
                "field": "query",
                "document": {
                    "message": "A new bonsai tree in the office"
                }
            }
        }
    }
    response = client.search(index_name, query)
    print(response)
    print()
    sleep(2.0)

    print("delete index:")
    response = client.delete_index(index_name)
    print(response)
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
