# tests/test_ortools.py
import numpy as np
import pytest
import scipy.sparse

ortools = pytest.importorskip("ortools.graph.python.min_cost_flow")

from sparse_ot.ortools_solver import solve_ortools


def _csr_from_dense(M):
    M = np.asarray(M, dtype=np.float64)
    n, m = M.shape
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    cols, costs = [], []
    for i in range(n):
        for j in range(m):
            cols.append(j)
            costs.append(M[i, j])
        row_ptr[i + 1] = len(cols)
    return row_ptr, np.asarray(cols, dtype=np.int32), np.asarray(costs, dtype=np.float64)


def test_ortools_2x2_optimal_routing():
    """Source 0→sink 0 and source 1→sink 1 is the unique optimum (cost 0)."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((2, 2))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan, np.diag([0.5, 0.5]), atol=1e-6)


def test_ortools_marginals_random():
    """Transport plan marginals match a and b within scaling tolerance."""
    rng = np.random.default_rng(0)
    n = 6
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((n, n))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


def test_ortools_cost_matches_pot():
    """OR-Tools transport cost matches POT within scaling tolerance."""
    import ot
    rng = np.random.default_rng(1)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    cost_ot = sum(M[r, c] * v for r, c, v in zip(rows, cols, vals))
    cost_pot = ot.emd2(a, b, M)
    rel_err = abs(cost_ot - cost_pot) / max(abs(cost_pot), 1e-15)
    assert rel_err < 1e-4, f"ortools={cost_ot}, pot={cost_pot}, rel_err={rel_err}"


def test_ortools_cost_scale_parameter_changes_precision():
    """Higher ortools_cost_scale yields a result no worse than lower scale."""
    import ot
    rng = np.random.default_rng(2)
    n = 6
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    cost_pot = ot.emd2(a, b, M)

    rows_low, cols_low, vals_low = solve_ortools(
        a, b, row_ptr, col_idx, costs, ortools_cost_scale=1e3
    )
    rows_high, cols_high, vals_high = solve_ortools(
        a, b, row_ptr, col_idx, costs, ortools_cost_scale=1e9
    )
    cost_low  = sum(M[r, c] * v for r, c, v in zip(rows_low, cols_low, vals_low))
    cost_high = sum(M[r, c] * v for r, c, v in zip(rows_high, cols_high, vals_high))
    err_low  = abs(cost_low  - cost_pot)
    err_high = abs(cost_high - cost_pot)
    assert err_high <= err_low + 1e-9


def test_ortools_rounding_residual_balanced():
    """Distributions that don't divide evenly into supply_scale still solve."""
    a = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    b = np.array([0.5, 0.25, 0.25])
    M = np.eye(3) * 0.0 + (1.0 - np.eye(3))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((3, 3))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


def test_ortools_returns_int32_indices_and_float64_values():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
    assert vals.dtype == np.float64
