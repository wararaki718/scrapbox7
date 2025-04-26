import hdbscan
import numpy as np
from sklearn.datasets import make_blobs, make_moons


def main() -> None:
    moons, _ = make_moons(n_samples=50, noise=0.05)
    blobs, _ = make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack((moons, blobs))
    print(f"test_data.shape: {test_data.shape}")
    print()

    model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        gen_min_span_tree=True,
    )
    model.fit(test_data)
    print(f"model.labels_: {model.labels_}")
    print(f"model.probabilities_: {model.probabilities_}")
    print(f"model.cluster_persistence_: {model.cluster_persistence_}")
    print(f"model.condensed_tree_: {model.condensed_tree_}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
