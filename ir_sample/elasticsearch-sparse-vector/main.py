from time import sleep

from client import ESClient
from loader import MappingLoader
from utils import get_docs, show


def main() -> None:
    host = "http://localhost:9200"
    client = ESClient(hosts=[host])

    filepath = "jsons/mappings.json"
    mappings = MappingLoader.load(filepath)

    print("create index:")
    index_name = "custom-index-2"
    response = client.create_index(index_name, mappings=mappings)
    print(response)
    print()
    sleep(2.0)

    print("data insert")
    for i, doc in enumerate(get_docs(), start=1):
        response = client.insert(index_name, i, doc)
    print(response)
    print()
    sleep(2.0)

    print("search(value=delicious):")
    query = {
        "query": {
            "term": {
                "impact": {
                    "value": "delicious",
                }
            }
        }
    }
    response = client.search(index_name, query)
    show(response)
    print()
    sleep(2.0)

    print("search(value=had):")
    query = {
        "query": {
            "term": {
                "impact": {
                    "value": "had",
                }
            }
        }
    }
    response = client.search(index_name, query)
    show(response)
    print()
    sleep(2.0)

    print("delete index:")
    response = client.delete_index(index_name)
    print(response)
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
