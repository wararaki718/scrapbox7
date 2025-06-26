from datasets import Dataset


class TextPreprocessor:
    def transform(self, data: Dataset, prompt: str) -> Dataset:
        data = data.map(
            lambda example: {
                "t5": f"{prompt} {example['text']}",
            }
        )
        return data
