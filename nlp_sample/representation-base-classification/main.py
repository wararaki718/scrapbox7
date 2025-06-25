from datasets import load_dataset

from classifier import TaskBaseClassifier, EmbeddingBaseClassifier, ZeroShotClassifier
from evaluation import evaluate

def main() -> None:
    data = load_dataset("rotten_tomatoes")
    print("---show dataset---")
    print(data)
    print()

    print("### task based classification ###")
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = TaskBaseClassifier(model_path)
    print("model loaded!\n")

    y_preds = model.estimate(data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print(performance)
    print()
    del model

    print("### embeddings based classification ###")
    model_path = "sentence-transformers/all-mpnet-base-v2"
    model = EmbeddingBaseClassifier(model_path)
    print("model loaded!\n")

    model.train(data["train"])
    y_preds = model.estimate(data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print(performance)
    print()
    del model

    print("### zero-shot classification ###")
    model_path = "sentence-transformers/all-mpnet-base-v2"
    model = ZeroShotClassifier(model_path)
    print("model loaded!\n")

    y_preds = model.estimate(data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print(performance)
    print()
    del model

    print("DONE")


if __name__ == "__main__":
    main()
