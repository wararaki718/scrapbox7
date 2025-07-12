from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


def load_model(
    model_name: str="openai/clip-vit-base-patch32",
) -> tuple[CLIPModel, CLIPTokenizer, CLIPProcessor]:
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    return (model, tokenizer, processor)
