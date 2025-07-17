from transformers import AutoTokenizer


class FormatPrompt:
    def __init__(self, model_name: str="TinyLlama/TinyLlama-1.1BChat-v1.0") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format(self, example: dict) -> dict:
        chat = example["chat"]
        prompt = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
        )

        return {"text": prompt}
