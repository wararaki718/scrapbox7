from fugashi import Tagger


class TextProcessor:
    def __init__(self):
        self._tagger = Tagger()
        self._stop_words = []

    def transform(self, text: str) -> list[str]:
        text = text.replace("。", "\n")
        sentences = text.split("\n")

        results = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence == "":
                continue

            tokens = []
            for token in self._tagger(sentence):
                feature = token.feature
                if feature.pos1 == "名詞" and token.surface not in self._stop_words:
                    tokens.append(token.surface)
            if tokens:
                results.append(tokens)

        return results
