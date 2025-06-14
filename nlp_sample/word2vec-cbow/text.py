import pickle
from collections import Counter
from pathlib import Path

from fugashi import Tagger
from tqdm import tqdm

from document import Document


class VocabularyGenerator:
    def generate(self, documents: list[Document], threshold: int=1) -> dict[str, int]:
        counter = Counter()
        for document in documents:
            counter.update(document.get_words())
        words = [word for word, total in counter.items() if total >= threshold]
        words.sort()
        vocabularies = {word: index for index, word in enumerate(words)}
        return vocabularies


class Tokenizer:
    def __init__(self, stop_words: list[str] | None = None) -> None:
        if stop_words is None:
            stop_words = []
        self._stop_words = stop_words
        self._tagger = Tagger()

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        for token in self._tagger(text):
            if token.feature.pos1 == "名詞" and token.surface not in self._stop_words:
                tokens.append(token.surface)
        return tokens


# in-memory processsing
class TextProcessor:
    def __init__(self, stop_words: list[str] | None = None) -> None:
        self._tokenizer = Tokenizer(stop_words)

    def transform(self, texts: list[str]) -> list[Document]:
        documents = []
        for text in tqdm(texts):
            text = text.replace("。", "\n")
            sentences = text.split("\n")

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence == "":
                    continue

                tokens = self._tokenizer.tokenize(sentence)
                if tokens:
                    documents.append(Document(tokens))
        return documents


class TextChunkProcessor:
    def __init__(self, stop_words: list[str] | None = None) -> None:
        self._tokenizer = Tokenizer(stop_words)

    def transform(self, texts: list[str], store_dir: Path = Path("./data"), chunksize: int=1024) -> None:
        documents = []
        index = 1

        for text in tqdm(texts):
            text = text.replace("。", "\n")
            sentences = text.split("\n")

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence == "":
                    continue

                tokens = self._tokenizer.tokenize(sentence)
                if tokens:
                    documents.append(Document(tokens))

            if len(documents) >= chunksize:
                store_path = store_dir / f"documents_{index:06d}.pkl"
                with open(store_path, "wb") as f:
                    pickle.dump(documents, f)
                print(f"Stored {len(documents)} documents to '{store_path}'", flush=True)
                documents = []
                index += 1


class TextChunkLoader:
    def __init__(self, store_dir: Path = Path("./data")) -> None:
        self._store_dir = store_dir

    def load(self) -> list[Document]:
        documents = []
        for file in sorted(self._store_dir.glob("documents_*.pkl")):
            with open(file, "rb") as f:
                chunk = pickle.load(f)
                documents.extend(chunk)
        return documents
