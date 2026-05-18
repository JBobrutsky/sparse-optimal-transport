import numpy as np
import pytest
import scipy.sparse

from sparse_ot._ext import _bonneel


def _csr(M):
    sp = scipy.sparse.csr_matrix(M)
    return (sp.indptr.astype(np.int32),
            sp.indices.astype(np.int32),
            sp.data.astype(np.float64),
            sp.shape[0], sp.shape[1])


def test_solve_sparse_full_support_matches_dense():
    rng = np.random.default_rng(0)
    n, m = 6, 8
    a = rng.dirichlet(np.ones(n)); a = a / a.sum()
    b = rng.dirichlet(np.ones(m)); b = b / b.sum()
    M = rng.uniform(0.0, 1.0, size=(n, m))

    row_ptr, col_idx, costs, _, _ = _csr(M)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    G_sp = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    G_d, u_d, v_d = _bonneel.solve_dense(a, b, M, 100000)

    np.testing.assert_allclose(np.sum(G_sp * M), np.sum(G_d * M), atol=1e-9)
    np.testing.assert_allclose(G_sp.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G_sp.sum(axis=0), b, atol=1e-9)


def test_solve_sparse_returns_duals():
    rng = np.random.default_rng(1)
    n, m = 5, 5
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))
    row_ptr, col_idx, costs, _, _ = _csr(M)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    assert u.shape == (n,)
    assert v.shape == (m,)
    G = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    primal = float(np.sum(G * M))
    dual = float(a @ u + b @ v)
    assert abs(primal - dual) < 1e-9


def test_solve_sparse_knn_support():
    rng = np.random.default_rng(2)
    n, m = 10, 10
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))

    k = 5  # k=4 produces an infeasible support for this seed; k=5 is feasible
    keep = np.argsort(M, axis=1)[:, :k]
    mask = np.zeros_like(M, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    M_sparse = np.where(mask, M, 0.0)
    sp = scipy.sparse.csr_matrix(M_sparse)
    sp.eliminate_zeros()
    row_ptr = sp.indptr.astype(np.int32)
    col_idx = sp.indices.astype(np.int32)
    costs = sp.data.astype(np.float64)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    G = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)


import resource
import sys


@pytest.mark.slow
def test_sparse_memory_scales_with_k():
    """10k x 10k with k=100k must fit comfortably under O(n*m) memory."""
    rng = np.random.default_rng(42)
    n = m = 10_000
    k_per_row = 10
    k = n * k_per_row

    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    a = a / a.sum()
    b = b / b.sum()

    cols = np.stack([
        rng.choice(m, size=k_per_row, replace=False) for _ in range(n)
    ], axis=0)
    cols.sort(axis=1)
    row_ptr = (np.arange(n + 1) * k_per_row).astype(np.int32)
    col_idx = cols.ravel().astype(np.int32)
    costs   = rng.uniform(0.0, 1.0, size=k).astype(np.float64)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows, cols_o, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 1_000_000
    )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is KB on Linux, bytes on macOS.
    bytes_per_unit = 1.0 if sys.platform == "darwin" else 1024.0
    delta_mb = (rss_after - rss_before) * bytes_per_unit / (1024.0 * 1024.0)
    assert delta_mb < 200, f"RSS grew by {delta_mb:.1f} MB, expected < 200 MB"


def test_default_num_iter_converges_at_16k():
    """n=16k, k=128 hit the old 100k-iter cap and silently returned bad flows.

    Regression test for the convergence cap bumping introduced after the mid
    benchmark surfaced marginal violations at ~1e-5.
    """
    from benchmarks.problems import generate_knn_grid_problem
    import sparse_ot
    a, b, M, _ = generate_knn_grid_problem(n=16_000, k=128, seed=0)
    G, info = sparse_ot.emd(a, b, M, log=True)
    fb_a = float(np.max(np.abs(np.asarray(G.sum(axis=1)).ravel() - a)))
    fb_b = float(np.max(np.abs(np.asarray(G.sum(axis=0)).ravel() - b)))
    assert max(fb_a, fb_b) < 1e-9
    assert info["result_code"] == 1
    assert info["warning"] is None


def test_warns_and_sets_result_code_on_truncation():
    """Passing an artificially low numItermax forces non-convergence; verify
    we emit RuntimeWarning and report result_code=0 with a populated warning."""
    import warnings as _warnings
    from benchmarks.problems import generate_knn_grid_problem
    import sparse_ot
    a, b, M, _ = generate_knn_grid_problem(n=16_000, k=128, seed=0)
    with _warnings.catch_warnings(record=True) as ws:
        _warnings.simplefilter("always")
        _, info = sparse_ot.emd(a, b, M, numItermax=100_000, log=True)
    assert info["result_code"] == 0
    assert info["warning"] is not None
    assert any(issubclass(w.category, RuntimeWarning) for w in ws)
