from pathlib import Path

from classifier import GeminiClassifier
from datasets import load_dataset

from evaluation import evaluate


def main() -> None:
    data = load_dataset("rotten_tomatoes")
    print("---show dataset---")
    print(data)
    print()

    # prompt
    model = GeminiClassifier(model="gemini-2.5-flash")
    prompt = Path("./prompt/sample.txt").read_text()

    # predict
    y_preds = model.estimate(prompt, data["test"])
    performance = evaluate(data["test"]["label"], y_preds)
    print(performance)
    print()
    del model

    print("DONE")


if __name__ == "__main__":
    main()
