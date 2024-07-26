from typing import List, Optional

from qdrant_client.http.models import Batch, ScoredPoint


def show(results: List[ScoredPoint]):
    for result in results:
        print(result)
    print()


def get_ids() -> List[int]:
    return list(range(1, 9))

def get_payloads() -> List[dict]:
    return [
        {"color": "green", "city": "London"},
        {"color": "red", "city": "London"},
        {"color": "red"},
        {"color": "blue"},
        {"color": None, "city": "Tokyo"},
        {"color": "yellow", "city": None},
        {"city": "Osaka"},
        {},
    ]


def get_vectors() -> List[Optional[dict]]:
    return [
        {"dense": [0.1, 0.9, 0.1]},
        {},
        {"dense": [0.1, 0.9, 0.1]},
        {"dense": [0.1, 0.9, 0.1]},
        {"dense": [0.1, 0.9, 0.1]},
        {"dense": [0.1, 0.9, 0.1]},
        {},
        {},
    ]
