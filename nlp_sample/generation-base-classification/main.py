from datasets import load_dataset

from classifier import T5Classifier
from evaluation import evaluate
from preprocess import TextPreprocessor


def main() -> None:
    data = load_dataset("rotten_tomatoes")
    print("---show dataset---")
    print(data)
    print()

    preprocessor = TextPreprocessor()
    prompt = "Is the following sentence positive or negative?"
    data = preprocessor.transform(data, prompt)

    print("### t5 classification ###")
    model_path = "google/flan-t5-small"
    model = T5Classifier(model_path)
    print("model loaded!\n")

    y_preds = model.estimate(data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print(performance)
    print()
    del model

    print("DONE")


if __name__ == "__main__":
    main()
