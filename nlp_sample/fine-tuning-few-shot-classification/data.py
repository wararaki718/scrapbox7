from datasets import load_dataset, Dataset
from setfit import sample_dataset


def get_data() -> tuple[Dataset, Dataset]:
    tomatoes = load_dataset("rotten_tomatoes")
    sample_data = sample_dataset(tomatoes["train"], num_samples=16)
    return sample_data, tomatoes["test"]
