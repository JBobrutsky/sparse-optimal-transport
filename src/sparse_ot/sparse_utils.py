# src/sparse_ot/sparse_utils.py
import warnings

import numpy as np
import scipy.sparse


_MARGINAL_TOL = 1e-6


def _default_num_iter(n, m, k):
    # Network simplex empirically converges in O((n+m) * sqrt(k)) pivots on
    # well-behaved OT problems. Pick a generous linear multiple of the problem
    # size so neither small nor large instances truncate. Capped to keep
    # pathological inputs from running unboundedly.
    return min(50_000_000, max(100_000, 100 * (n + m + k)))


def bonneel_sparse_solve(a, b, row_ptr, col_idx, costs, n, m, numItermax=None):
    """Run Bonneel's network simplex on a pre-built CSR system.

    Parameters
    ----------
    a, b        : float64 1-D arrays (marginals, already normalised).
    row_ptr, col_idx, costs : CSR arrays from ``to_csr``.
    n, m        : source / target sizes.
    numItermax  : pivot cap; ``None`` picks the problem-size-aware default.

    Returns
    -------
    G : CSR scipy matrix (n, m)
    u : float64 1-D, shape (n,)
    v : float64 1-D, shape (m,)
    """
    from sparse_ot._ext import _bonneel
    if numItermax is None:
        numItermax = _default_num_iter(n, m, len(costs))
    else:
        numItermax = int(numItermax)
    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, numItermax
    )
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    return G, u, v


_DEGENERATE_WARN_THRESHOLD = 0.05  # fraction of n+m-1


def bonneel_sparse_solve_warm(a, b, row_ptr, col_idx, costs, n, m,
                               G_warm, u0, v0, numItermax=None):
    """Warm-started network simplex: full basis (Mode B) or potential-only (Mode C).

    Mode B (full basis) is used when G_warm has enough non-degenerate arcs to
    reconstruct a spanning tree. Mode C (potential-only) is the fallback.

    Returns
    -------
    G    : CSR scipy matrix (n, m)
    u    : float64 1-D, shape (n,)
    v    : float64 1-D, shape (m,)
    warm_basis_used : bool — True if Mode B fired, False if Mode C fired.
    """
    from sparse_ot._ext import _bonneel

    if numItermax is None:
        numItermax = _default_num_iter(n, m, len(costs))
    else:
        numItermax = int(numItermax)

    n_basic = n + m - 1

    # 1. Drop stored zeros (inflated nnz from scipy CSR bookkeeping).
    G_warm = G_warm.copy()
    G_warm.eliminate_zeros()

    n_degenerate = n_basic - G_warm.nnz

    if n_degenerate < 0:
        # G_warm is non-basic (user-constructed, perturbed, etc.).
        warnings.warn(
            f"warm_start G has {-n_degenerate} more nonzeros than a spanning tree "
            f"(nnz={G_warm.nnz}, n+m-1={n_basic}). G_warm is not a basic feasible "
            "solution; arcs sorted by flow and spanning tree extracted.",
            RuntimeWarning, stacklevel=4,
        )
        n_degenerate = 0  # treat as non-degenerate; flow sort handles quality

    # 2. Mode selection.
    if n_degenerate > _DEGENERATE_WARN_THRESHOLD * n_basic:
        warnings.warn(
            f"warm_start G has {n_degenerate} zero-flow basic arcs "
            f"({100 * n_degenerate / n_basic:.0f}% of the spanning tree). "
            "Basis reconstruction will be approximate; using potential-only warm start.",
            RuntimeWarning, stacklevel=4,
        )
        warm_basis_used = False
        # Fall back to cold solve: injecting _pi after init() breaks the
        # artificial-star dual consistency (tree-arc rc ≠ 0), causing wrong
        # pivots. Mode B avoids this because warm tree arcs satisfy
        # complementary slackness with the prior-solve potentials.
        rows, cols, vals, u, v = _bonneel.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
    else:
        # Mode B — full basis. Sort arcs by flow descending so union-find
        # in C++ prefers high-flow (likely basic) arcs as tree arcs.
        coo = G_warm.tocoo()
        order = np.argsort(coo.data)[::-1]
        warm_rows = coo.row[order].astype(np.int32)
        warm_cols = coo.col[order].astype(np.int32)
        warm_flows = coo.data[order].astype(np.float64)

        # Compute CSR arc IDs for each warm arc via flat-key searchsorted.
        # row_ptr / col_idx are already sorted within each row (scipy guarantee).
        n_rows = n
        n_cols = m
        src_per_arc = np.repeat(
            np.arange(n_rows, dtype=np.int64), np.diff(row_ptr.astype(np.int64))
        )
        M_keys = src_per_arc * n_cols + col_idx.astype(np.int64)
        warm_keys = warm_rows.astype(np.int64) * n_cols + warm_cols.astype(np.int64)
        arc_ids = np.searchsorted(M_keys, warm_keys).astype(np.int32)

        warm_basis_used = True
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_basis(
            a, b, row_ptr, col_idx, costs,
            np.asarray(u0, dtype=np.float64),
            np.asarray(v0, dtype=np.float64),
            arc_ids, warm_rows, warm_cols, warm_flows,
            numItermax,
        )

    G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    return G, u, v, warm_basis_used


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
        # Do NOT call eliminate_zeros() here: zero-cost edges (e.g. self-edges
        # on a k-NN band with cost = (i-j)^2) are structurally required for
        # feasibility.  Removing them can turn a feasible instance infeasible.
        #
        # Filter per spec §3:
        #   1. Drop ±inf and NaN (absent-edge sentinels).
        #   2. Drop |cost| <= threshold when threshold > 0 (explicit sparsification).
        #      At threshold == 0.0 (the default), no costs are dropped — an
        #      explicit stored 0 is a real free edge.
        coo = csr.tocoo()
        if coo.data.size > 0:
            keep = np.isfinite(coo.data)
            if cost_sparsity_threshold > 0.0:
                keep &= np.abs(coo.data) > cost_sparsity_threshold
            if not keep.all():
                coo = scipy.sparse.coo_matrix(
                    (coo.data[keep], (coo.row[keep], coo.col[keep])),
                    shape=coo.shape,
                )
                csr = coo.tocsr()
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
