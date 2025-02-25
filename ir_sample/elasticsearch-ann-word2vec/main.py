from time import sleep

from client import ESClient
from loader import MappingLoader
from vectorizer import Word2VecVectorizer


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
    vectorizer = Word2VecVectorizer()
    # doc1
    text1 = "hello world test"
    vector1 = vectorizer.transform(text1)
    doc1 = {"custom_vector": vector1}

    print(f"add '{text1}'")
    response = client.insert(index_name, 1, doc1)
    print(response)

    # doc2
    text2 = "sample test vector"
    vector2 = vectorizer.transform(text2)
    doc2 = {"custom_vector": vector2}

    print(f"add '{text2}'")
    response = client.insert(index_name, 2, doc2)
    print(response)

    # doc3
    text3 = "hello word sample"
    vector3 = vectorizer.transform(text3)
    doc3 = {"custom_vector": vector3}

    print(f"add '{text3}'")
    response = client.insert(index_name, 3, doc3)
    print(response)
    print()
    sleep(2.0)

    # query
    print("search")
    keyword = "hello world"
    vector = vectorizer.transform(keyword)
    query = {
        "knn": {
            "query_vector": vector,
            "field": "custom_vector",
            "k": 3,
            "num_candidates": 5,
        },
        "_source": False,
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
