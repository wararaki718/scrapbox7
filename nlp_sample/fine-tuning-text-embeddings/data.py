from datasets import load_dataset
from torch.utils.data import Dataset


def get_data() -> tuple[Dataset, Dataset, list[float]]:
    # load dataset
    train_dataset = load_dataset(
        "glue", "mnli", split="train"
    ).select(range(50_000))
    train_dataset = train_dataset.remove_columns("idx")

    valid_dataset = load_dataset(
        "glue", "stsb", split="validation",
    )
    scores = [score / 5.0 for score in valid_dataset["label"]]

    return train_dataset, valid_dataset, scores
