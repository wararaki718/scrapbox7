from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def load_model(
    model_name: str = "bert-base-cased",
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
