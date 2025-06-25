import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import Pipeline
from transformers.pipelines.pt_utils import KeyDataset


def estimate(pipeline: Pipeline, testdata: Dataset) -> list[np.int64]:
    y_preds = []
    for output in tqdm(
        pipeline(KeyDataset(testdata, "text")), total=len(testdata),
    ):
        negative_score = output[0]["score"]
        positive_score = output[2]["score"]
        y_pred = np.argmax([positive_score, negative_score])
        y_preds.append(y_pred)

    return y_preds
