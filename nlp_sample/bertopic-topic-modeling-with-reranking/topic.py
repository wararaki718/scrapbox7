import umap
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer


def get_topic_model(model_name: str="thenlper/gte-small") -> BERTopic:
    bert_model = SentenceTransformer(model_name)
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
    return topic_model
