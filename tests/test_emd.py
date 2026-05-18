import numpy as np
import pytest
import ot
import scipy.sparse

import sparse_ot
from sparse_ot import emd
from sparse_ot.feasibility import InfeasibleProblemError


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


def test_emd_scipy_sparse_input():
    """scipy CSR cost matrix is accepted and returns scipy CSR transport plan."""
    rng = np.random.default_rng(99)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0, 1, (n, n))
    M_sp = scipy.sparse.csr_matrix(M_dense)
    G = sparse_ot.emd(a, b, M_sp)
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
    M_sp = scipy.sparse.csr_matrix(M_dense)
    cost_sot = sparse_ot.emd2(a, b, M_sp)
    cost_pot = ot.emd2(a, b, M_dense)
    assert abs(cost_sot - cost_pot) / abs(cost_pot) < 1e-6


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
        emd(a, b, M)


def test_emd_skips_check_for_dense_M():
    # Dense M (numpy ndarray) — feasibility check is skipped for dense input.
    # Use a fully non-zero cost matrix so no edges are dropped by to_csr.
    a = np.array([0.5, 0.5])
    b = np.array([0.3, 0.7])
    M = np.array([[0.5, 1.0], [1.0, 0.5]])
    emd(a, b, M)  # must not raise
