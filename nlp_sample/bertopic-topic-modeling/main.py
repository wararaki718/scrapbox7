import umap
from bertopic import BERTopic
from datasets import load_dataset
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer


def main() -> None:
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    abstracts = dataset["Abstracts"]
    print(f"Number of abstracts: {len(abstracts)}")

    bert_model = SentenceTransformer("thenlper/gte-small")
    umap_model = umap.UMAP(
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    topic_model = BERTopic(
        embedding_model=bert_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )
    print("model defined!")

    embeddings = bert_model.encode(abstracts, show_progress_bar=True)
    print(embeddings.shape)
    print()

    topic_model.fit(abstracts, embeddings)
    print("model fitted!")
    print()

    print("get topic info:")
    print(topic_model.get_topic_info())
    print()

    print("## find topics:")
    topics = topic_model.find_topics("deep learning")
    print(topics)
    print()

    print("## get topics:")
    result = topic_model.get_topic(22)
    print(result)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
