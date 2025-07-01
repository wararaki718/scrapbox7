from copy import deepcopy

from datasets import load_dataset

from evaluate import difference_topics
from topic import get_topic_model
from update import update_topic_model_keybert, update_topic_model_mrr


def main() -> None:
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    abstracts = dataset["Abstracts"]#[:500]
    print(f"Number of abstracts: {len(abstracts)}")

    topic_model = get_topic_model()
    print("model defined!")

    topic_model.fit(abstracts)
    print("model fitted!")
    print()

    original_topics = deepcopy(topic_model.topic_representations_)

    ## update topic with keybert
    topic_model = update_topic_model_keybert(topic_model, abstracts)
    df = difference_topics(topic_model, original_topics)
    print("## difference topics (keybert):")
    print(df)
    print()

    ## update topic with mrr
    topic_model = update_topic_model_mrr(topic_model, abstracts)
    df = difference_topics(topic_model, original_topics)
    print("## difference topics (mrr):")
    print(df)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
