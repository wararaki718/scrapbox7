from transformers import AutoTokenizer, AutoModelForMaskedLM


def load_models(model_name: str = "bert-base-cased") -> tuple[AutoTokenizer, AutoModelForMaskedLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model
