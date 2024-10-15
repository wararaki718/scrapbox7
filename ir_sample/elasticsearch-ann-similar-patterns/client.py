from typing import List

from elasticsearch import Elasticsearch


class ESClient:
    def __init__(self, hosts: List[str], timeout: float=30.0) -> None:
        self._client = Elasticsearch(hosts, timeout=timeout)

    def create_index(self, index_name: str, mappings: dict) -> dict:
        response = self._client.indices.create(index=index_name, body=mappings)
        return response

    def insert(self, index_name: str, id_: int, doc: dict) -> dict:
        response = self._client.create(index=index_name, id=id_, body=doc)
        return response

    def search(self, index_name: str, body: dict) -> dict:
        response = self._client.search(index=index_name, body=body)
        return response

    def delete_index(self, index_name: str) -> dict:
        response = self._client.indices.delete(index=index_name)
        return response
