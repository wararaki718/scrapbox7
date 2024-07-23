import numpy as np
from torchtext.datasets import AG_NEWS


class NewsLoader:
    @classmethod
    def load(cls, target: str="train", is_shuffle: bool=False) -> tuple[np.ndarray, np.ndarray]:
        if target == "train":
            data, _ = AG_NEWS()
        else:
            _, data = AG_NEWS()

        labels = []
        sentences = []
        for label, sentence in data:
            labels.append(label)
            sentences.append(sentence)
        labels = np.array(labels) - 1 # scaling
        sentences = np.array(sentences)

        if is_shuffle:
            indices = np.arange(labels.shape[0])
            np.random.shuffle(indices)
            sentences = sentences[indices]
            labels = labels[indices]

        return sentences, labels
