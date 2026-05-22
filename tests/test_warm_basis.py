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


def test_warm_basis_mode_b_same_support_same_metric():
    """Mode B: same support + same metric → cost matches cold, basis_used=True."""
    n = 50
    _, _, M = _band_csr(n, k=10, seed=1)
    # Slightly perturbed marginals avoid degenerate BFS so G_cold.nnz ≈ n+m-1,
    # which satisfies the ≥ 0.95*(n+m-1) threshold for Mode B dispatch.
    rng = np.random.default_rng(1)
    a = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    a = np.abs(a); a /= a.sum()
    b = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    b = np.abs(b); b /= b.sum()
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, nn, mm,
        G_cold, info["u"], info["v"],
    )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is True


def test_warm_basis_mode_b_same_support_perturbed_metric():
    """Mode B: same support, perturbed costs → result matches cold on perturbed M."""
    n = 50
    _, _, M1 = _band_csr(n, k=10, seed=2)
    # Slightly perturbed marginals avoid degenerate BFS so G1.nnz ≈ n+m-1,
    # which satisfies the ≥ 0.95*(n+m-1) threshold for Mode B dispatch.
    rng = np.random.default_rng(2)
    a = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    a = np.abs(a); a /= a.sum()
    b = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    b = np.abs(b); b /= b.sum()
    G1, info1 = emd(a, b, M1, log=True)

    rng = np.random.default_rng(42)
    M2_data = M1.data + rng.uniform(-0.01, 0.01, size=M1.nnz)
    M2 = scipy.sparse.csr_matrix(
        (M2_data, M1.indices.copy(), M1.indptr.copy()), shape=M1.shape
    )
    G2_cold, info2_cold = emd(a, b, M2, log=True)
    cold_cost2 = info2_cold["cost"]

    row_ptr, col_idx, costs2, nn, mm, _ = to_csr(M2, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs2, nn, mm,
        G1, info1["u"], info1["v"],
    )
    warm_cost = float(G_w.multiply(M2).sum())
    assert abs(warm_cost - cold_cost2) < 1e-9
    assert basis_used is True


def test_warm_basis_mode_b_subset_support():
    """Mode B: warm support ⊂ M_full support → matches cold solve on M_full."""
    n = 60
    _, _, M_full = _band_csr(n, k=12, seed=3)
    # Slightly perturbed marginals avoid degenerate BFS so G6.nnz ≈ n+m-1,
    # which satisfies the ≥ 0.95*(n+m-1) threshold for Mode B dispatch.
    rng = np.random.default_rng(3)
    a = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    a = np.abs(a); a /= a.sum()
    b = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    b = np.abs(b); b /= b.sum()

    half = 3
    rows6, cols6, costs6 = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + 6); lo = max(0, hi - 6)
        for j in range(lo, hi):
            rows6.append(i); cols6.append(j); costs6.append(float((i - j) ** 2))
    M6 = scipy.sparse.csr_matrix((costs6, (rows6, cols6)), shape=(n, n))
    G6, info6 = emd(a, b, M6, log=True)

    G_cold, info_cold = emd(a, b, M_full, log=True)
    cold_cost = info_cold["cost"]

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M_full, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, nn, mm,
        G6, info6["u"], info6["v"],
    )
    warm_cost = float(G_w.multiply(M_full).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is True


def test_warm_basis_sign_convention():
    """2x2 analytic problem: warm-start gives correct cost."""
    M = scipy.sparse.csr_matrix(np.array([[1., 2.], [3., 1.]]))
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    G_cold, info = emd(a, b, M, log=True, center_dual=False)
    cold_cost = info["cost"]
    assert abs(cold_cost - 1.0) < 1e-9

    row_ptr, col_idx, costs, n, m, _ = to_csr(M, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, n, m,
        G_cold, info["u"], info["v"],
    )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9


def test_warm_basis_non_basic_G_warns_and_correct():
    """Non-basic G_warm (nnz > n+m-1): warning emitted, result still correct."""
    import warnings
    n = 30
    a, b, M = _band_csr(n, k=8, seed=5)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    coo = G_cold.tocoo()
    # Pick a deterministic extra arc: row=1, col=2 is always in the k=8 band for n=30.
    extra_r, extra_c = 1, 2
    assert M[extra_r, extra_c] > 0, "extra arc not in M support — fix test setup"
    G_nonbasic = scipy.sparse.csr_matrix(
        (np.append(coo.data, 1e-15),
         (np.append(coo.row, extra_r), np.append(coo.col, extra_c))),
        shape=M.shape,
    )
    # nnz before solver (which calls eliminate_zeros internally)
    g_nonbasic_nnz = G_nonbasic.nnz

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G_w, u, v, basis_used = bonneel_sparse_solve_warm(
            a, b, row_ptr, col_idx, costs, nn, mm,
            G_nonbasic, info["u"], info["v"],
        )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    if g_nonbasic_nnz > n + n - 1:
        assert any("more nonzeros" in str(warning.message) for warning in w)
