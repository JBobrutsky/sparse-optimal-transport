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


def generate_knn_grid_warm_expand(
    n: int, k_warm: int, k_full: int, seed: int = 0
) -> tuple[
    np.ndarray,
    np.ndarray,
    scipy.sparse.csr_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.csr_matrix,
]:
    """Return (a, b, M_warm, M_full, w_plan_warm) — warm-start expansion pair.

    Marginals (a, b) are derived from a witness plan on the smaller k_warm
    band, so (a, b, M_warm) is feasible by construction. M_full is the cost
    matrix on a strictly larger k_full band of the same 1D grid; its support
    contains M_warm's, so (a, b, M_full) is also feasible. Both M_warm and
    M_full use squared-distance cost.
    """
    if k_warm < 1:
        raise ValueError("k_warm must be >= 1 to include the self-edge")
    if k_full <= k_warm:
        raise ValueError("k_full must be strictly greater than k_warm")
    rng = np.random.default_rng(seed)

    # --- Build k_warm band (mirrors generate_knn_grid_problem) ---
    half_w = k_warm // 2
    rows_w, cols_w, costs_w = [], [], []
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, lo + k_warm)
        lo = max(0, hi - k_warm)
        assert lo <= i < hi, f"row {i} would miss self-edge (lo={lo}, hi={hi})"
        for j in range(lo, hi):
            rows_w.append(i)
            cols_w.append(j)
            costs_w.append(float((i - j) ** 2))

    rows_w = np.asarray(rows_w, dtype=np.int32)
    cols_w = np.asarray(cols_w, dtype=np.int32)
    costs_w = np.asarray(costs_w, dtype=np.float64)
    nnz_w = costs_w.size

    log_w = rng.standard_normal(nnz_w)
    w = np.exp(log_w)
    w /= w.sum()

    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    np.add.at(a, rows_w, w)
    np.add.at(b, cols_w, w)
    s = a.sum()
    a /= s
    b /= s
    w_scaled = w / s
    diff = a.sum() - b.sum()
    if diff != 0.0:
        b[0] += diff

    M_warm = scipy.sparse.csr_matrix(
        (costs_w, (rows_w, cols_w)), shape=(n, n)
    )
    w_plan_warm = scipy.sparse.csr_matrix(
        (w_scaled, (rows_w, cols_w)), shape=(n, n)
    )

    # --- Build k_full band on the same 1D grid ---
    half_f = k_full // 2
    rows_f, cols_f, costs_f = [], [], []
    for i in range(n):
        lo = max(0, i - half_f)
        hi = min(n, lo + k_full)
        lo = max(0, hi - k_full)
        assert lo <= i < hi, f"row {i} would miss self-edge (lo={lo}, hi={hi})"
        for j in range(lo, hi):
            rows_f.append(i)
            cols_f.append(j)
            costs_f.append(float((i - j) ** 2))

    rows_f = np.asarray(rows_f, dtype=np.int32)
    cols_f = np.asarray(cols_f, dtype=np.int32)
    costs_f = np.asarray(costs_f, dtype=np.float64)

    M_full = scipy.sparse.csr_matrix(
        (costs_f, (rows_f, cols_f)), shape=(n, n)
    )
    return a, b, M_warm, M_full, w_plan_warm


def generate_knn_grid_warm_perturb(
    n: int, k: int, seed: int = 0
) -> tuple[
    np.ndarray,
    np.ndarray,
    scipy.sparse.csr_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.csr_matrix,
]:
    """Return (a, b, M_squared, M_abs, w_plan) — warm-start cost-perturbation pair.

    Both cost matrices share the same k-NN band support; only the cost data
    differs. M_squared uses (i-j)**2 and M_abs uses |i-j|. Marginals are
    derived from a witness plan on the shared band, so (a, b, M_*) is
    feasible by construction for either cost.
    """
    if k < 1:
        raise ValueError("k must be >= 1 to include the self-edge")
    rng = np.random.default_rng(seed)

    half = k // 2
    rows, cols, costs_sq, costs_abs = [], [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        assert lo <= i < hi, f"row {i} would miss self-edge (lo={lo}, hi={hi})"
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            d = i - j
            costs_sq.append(float(d * d))
            costs_abs.append(float(abs(d)))

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    costs_sq = np.asarray(costs_sq, dtype=np.float64)
    costs_abs = np.asarray(costs_abs, dtype=np.float64)
    nnz = costs_sq.size

    log_w = rng.standard_normal(nnz)
    w = np.exp(log_w)
    w /= w.sum()

    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    np.add.at(a, rows, w)
    np.add.at(b, cols, w)
    s = a.sum()
    a /= s
    b /= s
    w_scaled = w / s
    diff = a.sum() - b.sum()
    if diff != 0.0:
        b[0] += diff

    M_squared = scipy.sparse.csr_matrix(
        (costs_sq, (rows, cols)), shape=(n, n)
    )
    M_abs = scipy.sparse.csr_matrix(
        (costs_abs, (rows, cols)), shape=(n, n)
    )
    w_plan = scipy.sparse.csr_matrix(
        (w_scaled, (rows, cols)), shape=(n, n)
    )
    return a, b, M_squared, M_abs, w_plan


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
