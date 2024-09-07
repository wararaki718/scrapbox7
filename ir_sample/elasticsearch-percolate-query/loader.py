import json
from pathlib import Path


class MappingLoader:
    @classmethod
    def load(cls, filepath: Path) -> dict:
        with open(filepath) as f:
            mappings = json.load(f)
        return mappings
