from dataclasses import dataclass


@dataclass
class Document:
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def get_words(self) -> list[str]:
        return self._words
