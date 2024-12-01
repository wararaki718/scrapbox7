from dataset import TripletDataset
from sampler import RandomNegativeSampler, HardNegativeSampler
from utils import load_dummy_data


def main() -> None:
    print("load data:")
    impressions, X_queries, X_documents = load_dummy_data()
    print(len(impressions))
    print(X_queries.shape)
    print(X_documents.shape)
    print()

    print("pre-sampling:")
    random_sampler = RandomNegativeSampler()
    hard_sampler = HardNegativeSampler()

    random_indices = []
    hard_indices = []
    for impression in impressions:
        indices = random_sampler.sample(**impression)
        random_indices.extend(indices)
        indices = hard_sampler.sample(**impression, document_embeddings=X_documents)
        hard_indices.extend(indices)

    random_dataset = TripletDataset(random_indices, X_queries, X_documents)
    print(f"random dataset: {len(random_dataset)}")
    print(random_dataset._triplet_indices[:10])

    hard_dataset = TripletDataset(hard_indices, X_queries, X_documents)
    print(f"hard dataset: {len(hard_dataset)}")
    print(hard_dataset._triplet_indices[:10])
    print()

    print("DONE")


if __name__ == "__main__":
    main()
