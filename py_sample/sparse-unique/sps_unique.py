import numpy as np
import scipy.sparse as sps


def _unique(X: sps.csr_matrix, axis: int=0) -> tuple[sps.csr_matrix, np.ndarray]:
    """
    Returns a sparse matrix with the unique rows (axis=0)
    or columns (axis=1) of an input sparse matrix sp_matrix
    """
    if axis == 1:
        X = X.T

    old_format = X.getformat()
    dtype = np.dtype(X)
    ncols = X.shape[1]

    if old_format != 'lil':
        X = X.tolil()

    _, indices = np.unique(X.data + X.rows, return_index=True)
    rows = X.rows[indices]
    data = X.data[indices]
    nrows_unique = data.shape[0]

    X = sps.lil_matrix((nrows_unique, ncols), dtype=dtype)  #  or sp_matrix.resize(nrows_uniq, ncols)
    X.data = data
    X.rows = rows

    X = X.asformat(old_format)
    if axis == 1:
        X = X.T        
    return X, indices


def sps_unique(X: sps.csr_array, axis: int=0, keep_position: bool=False) -> tuple[sps.csr_matrix, np.ndarray]:
    X, indices = _unique(X, axis)
    if keep_position:
        if axis == 1:
            indices = np.lexsort(X.toarray()[::-1])
        else:
            indices = np.lexsort(X.T.toarray()[::-1])
        X = X[indices]
    return X, indices
