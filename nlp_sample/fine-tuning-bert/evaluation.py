import evaluate
import numpy as np
import torch


def compute_metrics(y_preds: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    logits, labels = y_preds
    predictions = np.argmax(logits, axis=1)
    
    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}
