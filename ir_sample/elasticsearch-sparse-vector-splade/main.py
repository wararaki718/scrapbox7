from time import sleep

from client import ESClient
from loader import MappingLoader
from utils import get_texts, show
from vectorizer import SpladeVectorizer


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


    print("vectorize texts:")
    vectorizer = SpladeVectorizer()
    texts = get_texts()
    vectors = vectorizer.transform(texts)
    print()

    print("create docs:")
    docs = []
    for text, vector in zip(texts, vectors):
        doc = {
            "text": text,
            "impact": vector,
        }
        docs.append(doc)
    print()

    print("data insert")
    for i, doc in enumerate(docs, start=1):
        response = client.insert(index_name, i, doc)
    print(response)
    print()
    sleep(2.0)

    print("search(value=hello):")
    query = {
        "query": {
            "term": {
                "impact": {
                    "value": "hello",
                }
            }
        }
    }
    response = client.search(index_name, query)
    show(response)
    sleep(2.0)

    print("search(value=dave):")
    query = {
        "query": {
            "term": {
                "impact": {
                    "value": "dave",
                }
            }
        }
    }
    response = client.search(index_name, query)
    show(response)
    sleep(2.0)

    print("delete index:")
    response = client.delete_index(index_name)
    print(response)
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
