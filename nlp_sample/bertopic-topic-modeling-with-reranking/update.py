from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance


def update_topic_model_keybert(
    topic_model: BERTopic,
    abstracts: list[str],
) -> BERTopic:
    representation_model = KeyBERTInspired()
    topic_model.update_topics(abstracts, representation_model=representation_model)

    return topic_model


def update_topic_model_mrr(
    topic_model: BERTopic,
    abstracts: list[str],
) -> BERTopic:
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    topic_model.update_topics(abstracts, representation_model=representation_model)

    return topic_model
