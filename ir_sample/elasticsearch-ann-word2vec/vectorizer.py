import numpy as np
from gensim import downloader


class Word2VecVectorizer:
    def __init__(self, model_name: str="word2vec-google-news-300") -> None:
        self._model = downloader.load(name=model_name)
    
    def transform(self, text: str) -> list[float]:
        words = []
        for word in text.split():
            if word in self._model:
                words.append(word)
        
        if not words:
            return np.zeros(self._model.model_size, dtype=np.float32).tolist()

        vector: list[float] = self._model[words].sum(axis=0).tolist()
        return vector
