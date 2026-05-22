"""Tests for warm-started network simplex entry points."""
import numpy as np
import scipy.sparse

from sparse_ot import emd
from sparse_ot.sparse_utils import bonneel_sparse_solve_warm, to_csr


def _band_csr(n, k, seed=0):
    rng = np.random.default_rng(seed)
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + k); lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i); cols.append(j)
            costs.append(float((i - j) ** 2) + rng.uniform(0, 0.1))
    M = scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )
    a = np.ones(n) / n
    b = np.ones(n) / n
    return a, b, M


def test_warm_potentials_mode_c_correct():
    """Mode C produces the same optimal cost as a cold solve."""
    import warnings
    n = 40
    a, b, M = _band_csr(n, k=8)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    # Construct a severely degenerate G_warm to force Mode C path.
    # Keep only 1 arc — n_degenerate >> 5% threshold.
    coo = G_cold.tocoo()
    idx = np.argmax(coo.data)
    G_degenerate = scipy.sparse.csr_matrix(
        ([coo.data[idx]], ([coo.row[idx]], [coo.col[idx]])), shape=M.shape
    )

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        G_warm, u, v, basis_used = bonneel_sparse_solve_warm(
            a, b, row_ptr, col_idx, costs, nn, mm,
            G_degenerate, info["u"], info["v"],
        )
    warm_cost = float(G_warm.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is False
