import numpy as np
import pytest
import scipy.sparse

import sparse_ot


def _band_problem(n, k, seed=0):
    """k-NN band problem on a 1D grid. Returns (a, b, M_csr)."""
    rng = np.random.default_rng(seed)
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            costs.append(float((i - j) ** 2))
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    costs = np.asarray(costs, dtype=np.float64)
    w = np.exp(rng.standard_normal(costs.size))
    w /= w.sum()
    a = np.zeros(n)
    b = np.zeros(n)
    np.add.at(a, rows, w)
    np.add.at(b, cols, w)
    a /= a.sum()
    b /= b.sum()
    M = scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )
    return a, b, M


def test_emd_accepts_warm_start_kwarg():
    """The warm_start kwarg exists. None is the default (cold path)."""
    a, b, M = _band_problem(20, 5)
    G = sparse_ot.emd(a, b, M, warm_start=None)
    assert G.shape == (20, 20)


def test_warm_start_with_dense_M_raises():
    """Dense M_full with warm_start is rejected per spec v1."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M_dense = np.array([[0.0, 1.0], [1.0, 0.0]])
    fake_warm = (np.eye(2) * 0.5, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError, match="sparse"):
        sparse_ot.emd(a, b, M_dense, warm_start=fake_warm)
