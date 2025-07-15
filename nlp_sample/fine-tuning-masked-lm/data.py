from datasets import load_dataset, Dataset


def get_data() -> tuple[Dataset, Dataset]:
    tomatoes = load_dataset("rotten_tomatoes")
    train_data = tomatoes["train"].remove_columns("label")
    test_data = tomatoes["test"].remove_columns("label")
    
    return train_data, test_data
