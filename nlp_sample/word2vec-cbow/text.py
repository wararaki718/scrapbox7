from collections import Counter

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
