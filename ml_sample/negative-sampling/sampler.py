import random
import torch


class RandomNegativeSampler:
    def sample(
        self,
        query_id: int,
        document_ids: torch.Tensor,
        actions: torch.Tensor,
        n_negative: int=2,
    ) -> list[tuple[int, int, int]]:
        positives = document_ids[actions==1]
        negatives: list[int] = document_ids[actions==0].tolist()
        indices = []
        for positive in positives:
            for negative in random.sample(negatives, k=n_negative):
                indices.append((query_id, positive.item(), negative))
        return indices


class HardNegativeSampler:
    def sample(
        self,
        query_id: int,
        document_ids: torch.Tensor,
        document_embeddings: torch.Tensor,
        actions: torch.Tensor,
        n_negative: int=2,
    ) -> list[tuple[int, int, int]]:
        positives = document_ids[actions==1]
        negatives = document_ids[actions==0]
        indices = []
        for positive in positives:
            similarities = torch.nn.functional.cosine_similarity(
                document_embeddings[positive],
                document_embeddings[negatives],
            )
            negatives = torch.argsort(similarities, descending=True)[:n_negative] # top-k
            for negative in negatives:
                indices.append((query_id, positive.item(), negative.item()))
        return indices
