import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


class TaskBaseClassifier:
    def __init__(self, model_path: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
        self._pipeline = pipeline(
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True,
            device="cpu",
        )

    def estimate(self, testdata: Dataset) -> list[np.int64]:
        y_preds = []
        for output in tqdm(
            self._pipeline(KeyDataset(testdata, "text")), total=len(testdata),
        ):
            negative_score = output[0]["score"]
            positive_score = output[2]["score"]
            y_pred = np.argmax([negative_score, positive_score])
            y_preds.append(y_pred)

        return y_preds


class EmbeddingBaseClassifier:
    def __init__(self, model_path: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self._model = SentenceTransformer(model_path)
        self._classifer = LogisticRegression(random_state=42)
    
    def train(self, traindata: Dataset) -> None:
        embeddings = self._model.encode(traindata["text"], show_progress_bar=True)
        self._classifer.fit(embeddings, traindata["label"])
    
    def estimate(self, testdata: Dataset) -> np.ndarray:
        embeddings = self._model.encode(testdata["text"], show_progress_bar=True)
        y_preds = self._classifer.predict(embeddings)
        return y_preds


class ZeroShotClassifier:
    def __init__(self, model_path: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self._model = SentenceTransformer(model_path)
    
    def estimate(self, testdata: Dataset) -> np.ndarray:
        embeddings = self._model.encode(testdata["text"], show_progress_bar=True)
        label_embeddings = self._model.encode(
            ["A negative review", "A positive review"], show_progress_bar=True
        )
        y_preds = cosine_similarity(embeddings, label_embeddings)
        y_preds = np.argmax(y_preds, axis=1)
        return y_preds
