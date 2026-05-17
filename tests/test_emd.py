import numpy as np
import pytest
import ot

import sparse_ot
from sparse_ot import emd


def _problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))
    return a, b, M


def test_emd_shape():
    a, b, M = _problem(5, 7)
    G = sparse_ot.emd(a, b, M)
    assert G.shape == (5, 7)


def test_emd_row_marginals():
    a, b, M = _problem(8, 6)
    G = sparse_ot.emd(a, b, M)
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)


def test_emd_col_marginals():
    a, b, M = _problem(8, 6)
    G = sparse_ot.emd(a, b, M)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)


def test_emd_matches_pot_rectangular():
    a, b, M = _problem(10, 12)
    G = sparse_ot.emd(a, b, M)
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd_matches_pot_square():
    a, b, M = _problem(20, 20)
    G = sparse_ot.emd(a, b, M)
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd2_matches_pot():
    a, b, M = _problem(15, 15)
    cost = sparse_ot.emd2(a, b, M)
    cost_ref = ot.emd2(a, b, M)
    assert abs(cost - cost_ref) / abs(cost_ref) < 1e-6


def test_emd2_consistent_with_emd():
    a, b, M = _problem(10, 10)
    G = sparse_ot.emd(a, b, M)
    cost_from_plan = float(np.sum(G * M))
    cost_from_emd2 = sparse_ot.emd2(a, b, M)
    assert abs(cost_from_plan - cost_from_emd2) < 1e-12


def test_emd_solver_override_bonneel():
    a, b, M = _problem(8, 8)
    G = sparse_ot.emd(a, b, M, solver="bonneel")
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd_accepts_cost_sparsity_threshold():
    # Threshold parameter is accepted without error in Plan 1.
    # Routing behavior based on threshold comes in Plan 2.
    a, b, M = _problem(5, 5)
    G = sparse_ot.emd(a, b, M, cost_sparsity_threshold=0.01)
    assert G.shape == (5, 5)


def test_emd2_return_matrix():
    a, b, M = _problem(5, 5)
    cost, G = sparse_ot.emd2(a, b, M, return_matrix=True)
    assert isinstance(cost, float)
    assert G.shape == (5, 5)


def test_emd_log():
    a, b, M = _problem(5, 5)
    G, log = sparse_ot.emd(a, b, M, log=True)
    assert G.shape == (5, 5)
    assert isinstance(log, dict)


def test_emd_nonnegative_transport():
    a, b, M = _problem(10, 10)
    G = sparse_ot.emd(a, b, M)
    assert np.all(G >= -1e-12)


# --- LEMON path tests ---

def test_emd_lemon_override():
    a, b, M = _problem(8, 8)
    G = sparse_ot.emd(a, b, M, solver='lemon')
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd2_lemon_override():
    a, b, M = _problem(10, 10)
    cost = sparse_ot.emd2(a, b, M, solver='lemon')
    cost_ref = ot.emd2(a, b, M)
    assert abs(cost - cost_ref) / abs(cost_ref) < 1e-6


def test_emd_scipy_sparse_input():
    """scipy CSR cost matrix is accepted and returns scipy CSR transport plan."""
    rng = np.random.default_rng(99)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0, 1, (n, n))
    import scipy.sparse
    M_sp = scipy.sparse.csr_matrix(M_dense)
    G = sparse_ot.emd(a, b, M_sp, solver='lemon')
    assert scipy.sparse.issparse(G)
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=1)).ravel(), a, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=0)).ravel(), b, atol=1e-9
    )


def test_emd2_scipy_sparse_input():
    """emd2 with scipy CSR input returns same cost as POT."""
    rng = np.random.default_rng(55)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0, 1, (n, n))
    import scipy.sparse
    M_sp = scipy.sparse.csr_matrix(M_dense)
    cost_sot = sparse_ot.emd2(a, b, M_sp, solver='lemon')
    cost_pot = ot.emd2(a, b, M_dense)
    assert abs(cost_sot - cost_pot) / abs(cost_pot) < 1e-6


def test_emd_invalid_solver_raises():
    a, b, M = _problem(5, 5)
    with pytest.raises(ValueError):
        sparse_ot.emd(a, b, M, solver='invalid')


def test_emd_cost_sparsity_threshold_drops_edges():
    """cost_sparsity_threshold drops low-cost edges; result has correct marginals."""
    rng = np.random.default_rng(11)
    n = 10
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0, 1, (n, n))
    G = sparse_ot.emd(a, b, M, cost_sparsity_threshold=0.2)
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)


def test_emd_ortools_override_dense():
    """solver='ortools' on a dense problem matches POT cost to 1e-4."""
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    a, b, M = _problem(8, 8)
    cost_sot = sparse_ot.emd2(a, b, M, solver='ortools')
    cost_pot = ot.emd2(a, b, M)
    rel_err = abs(cost_sot - cost_pot) / abs(cost_pot)
    assert rel_err < 1e-4


def test_emd_ortools_scipy_sparse_input():
    """solver='ortools' with scipy CSR cost matrix returns scipy CSR plan."""
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    import scipy.sparse
    rng = np.random.default_rng(123)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0.1, 1.0, (n, n))
    M_sp = scipy.sparse.csr_matrix(M_dense)
    G = sparse_ot.emd(a, b, M_sp, solver='ortools')
    assert scipy.sparse.issparse(G)
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=1)).ravel(), a, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=0)).ravel(), b, atol=1e-6
    )


def test_emd_ortools_cost_scale_passes_through():
    """ortools_cost_scale parameter reaches the solver."""
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    a, b, M = _problem(6, 6)
    cost_default = sparse_ot.emd2(a, b, M, solver='ortools')
    cost_higher  = sparse_ot.emd2(a, b, M, solver='ortools',
                                   ortools_cost_scale=1e9)
    assert np.isfinite(cost_default) and np.isfinite(cost_higher)


import scipy.sparse
from sparse_ot.feasibility import InfeasibleProblemError


def test_emd_raises_on_infeasible_sparse_support():
    # Singleton components: source 0 has 0.5 mass, but only edge (0,0) exists
    # and target 0 has only 0.1 mass demand. Imbalanced component → infeasible.
    rows = [0, 1, 1, 2, 2]
    cols = [0, 1, 2, 1, 2]
    data = [1.0] * 5
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
    a = np.array([0.5, 0.25, 0.25])
    b = np.array([0.1, 0.45, 0.45])
    with pytest.raises(InfeasibleProblemError):
        emd(a, b, M, solver='lemon')


def test_emd_skips_check_for_dense_M():
    # Dense M (numpy ndarray) — feasibility check is skipped for dense input.
    # Use a fully non-zero cost matrix so no edges are dropped by to_csr.
    a = np.array([0.5, 0.5])
    b = np.array([0.3, 0.7])
    M = np.array([[0.5, 1.0], [1.0, 0.5]])
    emd(a, b, M)  # must not raise


def test_emd_bonneel_on_sparse_raises():
    # Bonneel cannot tell "absent edge" from "real edge with cost 0" once a
    # sparse M is materialized via toarray() — silently returns degenerate
    # zero-cost plans. Refuse the combination at the API boundary.
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M_sp = scipy.sparse.csr_matrix(np.array([[0.5, 1.0], [1.0, 0.5]]))
    with pytest.raises(ValueError, match="Bonneel.*sparse"):
        emd(a, b, M_sp, solver='bonneel')
