from datasets import Dataset, load_dataset

from prompt import FormatPrompt

def get_data(prompt: FormatPrompt, n_data: int=3000) -> Dataset:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    dataset = dataset.shuffle(seed=42).select(range(n_data))
    dataset = dataset.map(prompt.format)
    return dataset
