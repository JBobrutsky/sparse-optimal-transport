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
