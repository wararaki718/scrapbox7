import numpy as np


def _unpack_tuple(x: tuple) -> tuple | np.ndarray:
    if len(x) == 1:
        return x[0]
    else:
        return x


def _unique1d(
    ar: np.ndarray,
    return_index: bool | None=False,
    return_inverse: bool | None=False,
    return_counts: bool | None=False,
    axis: int | None=None,
    equal_nan: bool | None=True,
    inverse_shape: tuple | None=None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    # ar = np.asanyarray(ar)#.flatten()
    ar = ar.reshape((1, -1))[0]

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    # mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True
    if (equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and
            np.isnan(aux[-1])):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = (
                aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx.reshape(inverse_shape) if axis is None else inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


def np_unique(
    x: np.ndarray,
    return_index: bool | None=False,
    return_inverse: bool | None=False,
    return_counts: bool | None=False,
    axis: int | None=None,
    equal_nan: bool | None=True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ar = np.asanyarray(x)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts, 
                        equal_nan=equal_nan, inverse_shape=ar.shape, axis=None)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.exceptions.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.exceptions.AxisError(axis, ar.ndim) from None
    inverse_shape = [1] * ar.ndim
    inverse_shape[axis] = ar.shape[0]

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(
        consolidated,
        return_index,
        return_inverse,
        return_counts,
        equal_nan=equal_nan, inverse_shape=inverse_shape,
        axis=axis
    )
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)
