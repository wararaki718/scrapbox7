from datasets import load_dataset, Dataset


def get_data() -> tuple[Dataset, Dataset]:
    tomatoes = load_dataset("rotten_tomatoes")
    
    return tomatoes["train"], tomatoes["test"]
