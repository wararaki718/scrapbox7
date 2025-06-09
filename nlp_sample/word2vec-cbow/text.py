from collections import Counter

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


class TextPreprocessor:
    def transform(self, texts: list[str]) -> list[Document]:
        documents = []
        for text in texts:
            text = text.lower()
            text = text.replace('.', '').replace(',', '')
            words = text.split()
            documents.append(Document(words))

        return documents
