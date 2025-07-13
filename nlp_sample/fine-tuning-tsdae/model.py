from sentence_transformers import SentenceTransformer, models


def load_model(model_name: str = "bert-base-uncased") -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model
