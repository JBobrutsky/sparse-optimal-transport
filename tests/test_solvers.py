"""Unit tests for benchmarks/solvers.py adapter functions."""
import numpy as np
import pytest
import scipy.sparse


def _two_by_two():
    """2×2 OT problem with known optimal cost = 1.0."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[1.0, 2.0], [3.0, 1.0]])
    return a, b, M


def test_solve_result_fields():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    r = solve_sparse_ot(a, b, M)
    assert hasattr(r, "cost")
    assert hasattr(r, "wall_s")
    assert hasattr(r, "peak_mb")
    assert hasattr(r, "marginal_err_a")
    assert hasattr(r, "marginal_err_b")


def test_sparse_ot_dense_correct():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    r = solve_sparse_ot(a, b, M)
    assert abs(r.cost - 1.0) < 1e-9
    assert r.marginal_err_a < 1e-10
    assert r.marginal_err_b < 1e-10
    assert r.wall_s > 0
    assert r.peak_mb >= 0


def test_sparse_ot_csr_correct():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    M_csr = scipy.sparse.csr_matrix(M)
    r = solve_sparse_ot(a, b, M_csr)
    assert abs(r.cost - 1.0) < 1e-9



def test_pot_agrees_with_sparse_ot():
    from benchmarks.solvers import solve_sparse_ot, solve_pot
    a, b, M = _two_by_two()
    r_ot = solve_sparse_ot(a, b, M)
    r_pot = solve_pot(a, b, M)
    assert r_pot is not None
    assert abs(r_pot.cost - r_ot.cost) < 1e-10
    assert r_pot.marginal_err_a < 1e-10
    assert r_pot.marginal_err_b < 1e-10


def test_pot_skips_large_n():
    from benchmarks.solvers import solve_pot, POT_MAX_N
    n = POT_MAX_N + 1
    # Diagonal CSR: n edges but n > POT_MAX_N triggers skip.
    M = scipy.sparse.eye(n, format="csr")
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_pot(a, b, M) is None


def test_pot_skips_large_nnz():
    from benchmarks.solvers import solve_pot, POT_MAX_NNZ
    # 317×317 = 100489 > 100000.
    n = 317
    M = scipy.sparse.csr_matrix(np.ones((n, n)))
    assert M.nnz > POT_MAX_NNZ
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_pot(a, b, M) is None


def test_ortools_agrees_with_sparse_ot():
    from benchmarks.solvers import solve_sparse_ot, solve_ortools
    a, b, M = _two_by_two()
    r_ot = solve_sparse_ot(a, b, M)
    r_or = solve_ortools(a, b, M)
    assert r_or is not None
    # OR-Tools rounds costs to 1/SCALE; expect ~1e-6 relative accuracy.
    assert abs(r_or.cost - r_ot.cost) < 1e-5
    assert r_or.marginal_err_a < 2e-6
    assert r_or.marginal_err_b < 2e-6


def test_ortools_skips_large_nnz():
    from benchmarks.solvers import solve_ortools, ORTOOLS_MAX_NNZ
    # 800×800 = 640000 > 500000.
    n = 800
    M = scipy.sparse.csr_matrix(np.ones((n, n)))
    assert M.nnz > ORTOOLS_MAX_NNZ
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_ortools(a, b, M) is None


def test_ortools_skips_large_n():
    from benchmarks.solvers import solve_ortools, ORTOOLS_MAX_N
    n = ORTOOLS_MAX_N + 1
    M = scipy.sparse.eye(n, format="csr")
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_ortools(a, b, M) is None
