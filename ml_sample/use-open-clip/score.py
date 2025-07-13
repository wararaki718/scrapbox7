import numpy as np
import torch


def similarity_score(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> np.ndarray:
    # normalize the embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    # score
    text_embeddings = text_embeddings.detach().cpu().numpy()
    image_embeddings = image_embeddings.detach().cpu().numpy()
    score = np.dot(text_embeddings, image_embeddings.T)

    return score
