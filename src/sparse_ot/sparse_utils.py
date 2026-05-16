# src/sparse_ot/sparse_utils.py
import numpy as np
import scipy.sparse


def to_csr(M, cost_sparsity_threshold=0.0):
    """Convert a cost matrix to CSR format for the C++ solvers.

    For scipy sparse input, converts to CSR and casts indices/data; threshold
    is NOT applied (the caller controls sparsity via the scipy matrix itself).
    For dense numpy input, entries with |M[i,j]| <= threshold are dropped.
    Exact zeros are always dropped from dense input.

    Returns
    -------
    row_ptr : int32 ndarray, shape (n+1,)
    col_idx : int32 ndarray, shape (nnz,)
    costs   : float64 ndarray, shape (nnz,)
    n, m    : int — source and target sizes
    nnz     : int — number of edges
    """
    if scipy.sparse.issparse(M):
        csr = M.tocsr().astype(np.float64)
        csr.eliminate_zeros()
        n, m = csr.shape
        return (
            csr.indptr.astype(np.int32),
            csr.indices.astype(np.int32),
            np.asarray(csr.data, dtype=np.float64),
            n, m, int(csr.nnz),
        )

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    n, m = M.shape

    mask = np.abs(M) > cost_sparsity_threshold  # excludes exact zeros and <= threshold

    row_counts = mask.sum(axis=1).astype(np.int32)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(row_counts, out=row_ptr[1:])
    nnz = int(row_ptr[n])

    col_idx = np.empty(nnz, dtype=np.int32)
    costs = np.empty(nnz, dtype=np.float64)
    for i in range(n):
        s, e = int(row_ptr[i]), int(row_ptr[i + 1])
        js = np.where(mask[i])[0].astype(np.int32)
        col_idx[s:e] = js
        costs[s:e] = M[i, js]

    return row_ptr, col_idx, costs, n, m, nnz
