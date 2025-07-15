from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline


def evaluate(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, text: str) -> None:
    # Create a fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # Use the pipeline to predict masked tokens
    results = fill_mask(text)

    # Print the results
    for result in results:
        print(f">>> {result['sequence']}")
