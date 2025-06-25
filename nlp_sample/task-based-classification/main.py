from datasets import load_dataset
from transformers import pipeline

from estimation import estimate
from evaluation import evaluate

def main() -> None:
    data = load_dataset("rotten_tomatoes")
    print("---show dataset---")
    print(data)
    print()

    print(type(data["test"]))

    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device="cpu",
    )
    print("model defiled!")
    print()

    y_preds = estimate(pipe, data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print()
    print(performance)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
