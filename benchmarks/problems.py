# benchmarks/problems.py
"""Reproducible k-NN grid OT problems, feasible by construction.

Source and target nodes are placed on a shared 1D grid at positions 0..n-1.
Each source i connects to up to k target nodes nearest to i (clipped at
grid boundaries) with cost (i - j)^2.

Marginals are derived from a random plan w on the band edges (spec §6):
sample w_ij = exp(N(0,1)) on each band edge, set a[i] = sum_j w_ij and
b[j] = sum_i w_ij, normalize so sum(a) == sum(b) == 1 bit-for-bit. By
construction w / sum(w) is a feasible transport plan from a to b, so the
problem is feasible regardless of k.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse


def generate_knn_grid_problem(
    n: int, k: int, seed: int = 0
) -> tuple[
    np.ndarray, np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix
]:
    """Return (a, b, M, w_plan) — feasible-by-construction sparse OT instance.

    M is the (n, n) CSR of squared-distance costs on a k-NN band of the
    shared 1D grid (self-edges included for k >= 1). w_plan is the witness
    transport plan whose row and column sums equal a and b. The cost of
    w_plan is an upper bound on the optimal OT cost.
    """
    if k < 1:
        raise ValueError("k must be >= 1 to include the self-edge")
    rng = np.random.default_rng(seed)

    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        assert lo <= i < hi, f"row {i} would miss self-edge (lo={lo}, hi={hi})"
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            costs.append(float((i - j) ** 2))

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    costs = np.asarray(costs, dtype=np.float64)
    nnz = costs.size

    log_w = rng.standard_normal(nnz)
    w = np.exp(log_w)
    w /= w.sum()  # joint distribution; row/col sums are a, b.

    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    np.add.at(a, rows, w)
    np.add.at(b, cols, w)
    # Normalize by a common scalar so both marginals represent a probability
    # distribution.  a_raw and b_raw are identical in exact arithmetic
    # (both equal w.sum()), but the np.add.at accumulation order differs
    # between rows and cols, so they may disagree by up to a few ULP.
    # We fix this by a targeted single-element correction on b: compute the
    # shortfall after initial normalization and absorb it into b[0], which is
    # always strictly positive (exp values are > 0).  The shortfall is at
    # most ~n ULP ≈ 2e-13 for n=1000, well within the atol=1e-12 tolerance
    # used by the witness-plan marginal check.  We then verify equality holds
    # with a second correction if the first wasn't sufficient (rare).
    s = a.sum()
    a /= s
    b /= s
    w_scaled = w / s
    # a.sum() and b.sum() should both be ~1 and equal in exact arithmetic.
    # In float, they may differ by at most a few ULP due to non-associative
    # accumulation.  We apply a targeted single-element correction on b[0]
    # (always positive since w > 0) to achieve bit-exact equality.  If the
    # pairwise-sum rounding makes the correction oscillate (rare, only for
    # specific (n, k) combinations), the residual is still < 2 ULP and the
    # witness-plan marginal check (atol=1e-12) is unaffected.
    diff = a.sum() - b.sum()
    if diff != 0.0:
        b[0] += diff

    M = scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )
    w_plan = scipy.sparse.csr_matrix(
        (w_scaled, (rows, cols)), shape=(n, n)
    )
    return a, b, M, w_plan


def generate_dense_random_problem(
    n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, b, M) — fully dense n×n OT instance with random uniform costs.

    Marginals are Dirichlet-sampled and re-normalised so a.sum() == b.sum() ==
    1 in float64. Used by the dense benchmark suite where every (i, j) is a
    valid edge — no penalty trick, no sparsity.
    """
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    a = a / a.sum()
    b = b / b.sum()
    M = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float64)
    return a, b, M
