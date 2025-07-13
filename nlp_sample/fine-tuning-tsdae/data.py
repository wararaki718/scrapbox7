from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers.datasets import DenoisingAutoEncoderDataset


def get_data() -> tuple[Dataset, Dataset, list[float]]:
    # load dataset
    mnli = load_dataset("glue", "mnli", split="train").select(range(25_000))
    valid_dataset = load_dataset("glue", "stsb", split="validation")

    # preprocess
    flat_sentences = mnli["premise"] + mnli["hypothesis"]
    damaged_data = DenoisingAutoEncoderDataset(list(set(flat_sentences)))

    # create data
    train_dataset = {"damaged_sentence": [], "original_sentence": []}
    for data in tqdm(damaged_data):
        train_dataset["damaged_sentence"].append(data.texts[0])
        train_dataset["original_sentence"].append(data.texts[1])
    
    valid_scores = [score / 5 for score in valid_dataset["label"]]
    
    return Dataset.from_dict(train_dataset), valid_dataset, valid_scores
