import numpy as np
import pytest
import ot

import sparse_ot


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
