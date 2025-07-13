from model import load_model
from score import similarity_score
from utils import get_image


def main() -> None:
    image = get_image()
    caption = "a puppy playing in the snow"

    model, tokenizer, processor = load_model()
    inputs = tokenizer(caption, return_tensors="pt")
    text_embeddings = model.get_text_features(**inputs)
    print(text_embeddings.shape)
    print()

    processed_image = processor(
        text=None, images=image, return_tensors="pt",
    )["pixel_values"]
    image_embeddings = model.get_image_features(processed_image)
    print(image_embeddings.shape)
    print()

    score = similarity_score(text_embeddings, image_embeddings)
    print(f"score: {score}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
