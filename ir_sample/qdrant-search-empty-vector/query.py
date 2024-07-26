from dataclasses import dataclass, asdict
from typing import Dict, List

from qdrant_client.http.models import Filter, NamedVector, SearchParams


@dataclass
class SearchQuery:
    query_filter: Filter
    search_params: SearchParams
    query_vector: NamedVector

    def to_dict(self) -> dict:
        return asdict(self)
