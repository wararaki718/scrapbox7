from qdrant_client.models import VectorParams, PointStruct

from builder import QueryBuilder
from client import SearchClient
from condition import SearchCondition
from utils import get_ids, get_payloads, get_vectors, show


def main():
    collection_name = "sample"
    dim = 3

    client = SearchClient()
    params = VectorParams(
        size=dim,
        distance="Cosine"
    )
    _ = client.create_index(collection_name, params)
    print(f"index created: {collection_name}")

    points = []
    for point_id, payload, vector in zip(get_ids(), get_payloads(), get_vectors()):
        point = PointStruct(
            id=point_id,
            payload=payload,
            vector=vector,
        )
        points.append(point)
    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()

    # prefilter
    condition = SearchCondition(city="London")
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(color="red")
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition()
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(color="yellow", is_empty_city=True)
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(is_empty_city=True)
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(is_empty_color=True)
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(city="London", is_empty_city=True)
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    condition = SearchCondition(is_empty_color=True, is_empty_city=True)
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
