from bertopic import BERTopic
from bertopic.representation import TextGeneration
from transformers import pipeline


def update_topic_model(
    topic_model: BERTopic,
    abstracts: list[str],
    prompt: str,
) -> BERTopic:
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    representation_model = TextGeneration(
        generator,
        prompt=prompt,
        doc_length=50,
        tokenizer="whitespace",
    )

    topic_model.update_topics(
        abstracts,
        representation_model=representation_model,
    )
    return topic_model
