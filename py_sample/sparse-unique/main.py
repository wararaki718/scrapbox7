import numpy as np
import scipy.sparse as sps

from sps_unique import sps_unique


def main() -> None:
    # Test
    # Create a large sparse matrix with elements in [0, 10]
    A = 10 * sps.random(10000, 3, 0.5, format='csr')
    #A = np.ceil(A).astype(int)

    # unique rows
    A_uniq, indices = sps_unique(A, axis=0)
    A_uniq_numpy = np.unique(A.toarray(), axis=0)
    print(indices)
    print(A_uniq.shape)
    print(A_uniq_numpy.shape)
    print()

    # keep position
    A_uniq, indices = sps_unique(A, axis=0, keep_position=True)
    print(indices)
    print(A_uniq.toarray().shape)
    print(A_uniq_numpy.shape)
    print()

    # unique columns
    A_uniq, indices = sps_unique(A, axis=1)
    A_uniq_numpy = np.unique(A.toarray(), axis=1)
    print(indices)
    print(A_uniq.shape)
    print(A_uniq_numpy.shape)
    print()


if __name__ == '__main__':    
    main()
