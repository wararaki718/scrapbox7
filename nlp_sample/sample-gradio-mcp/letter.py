class LetterCounter:
    def count(self, text: str, letter: str) -> int:
        text = text.lower()
        return text.count(letter)
