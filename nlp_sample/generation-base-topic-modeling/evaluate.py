import numpy as np
import pandas as pd
from bertopic import BERTopic


def difference_topics(
    model: BERTopic,
    original_topics: np.ndarray,
    n_topics: int=5,
    n_words: int=5,
) -> pd.DataFrame:
    original_topic_words = []
    new_topic_words = []
    topics = []
    for i in range(n_topics):
        original_words = " | ".join(list(zip(*original_topics[i]))[0][:n_words])
        new_words = " | ".join(list(zip(*model.get_topic(i)))[0][:n_words])

        topics.append(i)
        original_topic_words.append(original_words)
        new_topic_words.append(new_words)

    df = pd.DataFrame({
        "Topic": topics,
        "Original": original_topic_words,
        "Updated": new_topic_words,
    })
    return df
