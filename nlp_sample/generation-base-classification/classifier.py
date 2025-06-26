from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


class T5Classifier:
    def __init__(self, model_path: str = "google/flan-t5-small") -> None:
        self._pipeline = pipeline(
            "text2text-generation",
            model=model_path,
            device="cpu",
        )

    def estimate(self, testdata: Dataset) -> list[int]:
        y_preds = []
        for output in tqdm(
            self._pipeline(KeyDataset(testdata, "text")), total=len(testdata),
        ):
            text = output[0]["generated_text"]
            y_pred = int(text == "negative")
            y_preds.append(y_pred)

        return y_preds
