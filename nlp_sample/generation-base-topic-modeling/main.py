from copy import deepcopy
from pathlib import Path

from datasets import load_dataset

from evaluate import difference_topics
from topic import get_topic_model
from update import update_topic_model


def main() -> None:
    # load dataset
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    abstracts = dataset["Abstracts"]
    print(f"Number of abstracts: {len(abstracts)}")

    # modeling
    model = get_topic_model()
    model.fit(abstracts)
    print("model defined!")

    original_topics = deepcopy(model.topic_representations_)

    # generate base
    prompt = Path("prompt/sample.txt").read_text()
    model = update_topic_model(
        topic_model=model,
        abstracts=abstracts,
        prompt=prompt,
    )
    print("model updated!")

    df = difference_topics(model, original_topics)
    print("## difference topics:")
    print(df)

    print("DONE")


if __name__ == "__main__":
    main()
