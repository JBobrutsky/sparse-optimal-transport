import numpy as np
import pytest
import scipy.sparse

import sparse_ot


def _problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))
    return a, b, M


def test_dense_log_dict_keys():
    a, b, M = _problem(6, 8)
    G, info = sparse_ot.emd(a, b, M, log=True)
    assert set(info.keys()) >= {"cost", "u", "v", "warning", "result_code"}
    assert info["u"].shape == (6,)
    assert info["v"].shape == (8,)


def test_dense_strong_duality():
    a, b, M = _problem(7, 9)
    G, info = sparse_ot.emd(a, b, M, log=True)
    primal = float(np.sum(G * M))
    dual = float(a @ info["u"] + b @ info["v"])
    assert abs(primal - dual) < 1e-9


def test_dense_dual_feasibility():
    a, b, M = _problem(7, 9)
    G, info = sparse_ot.emd(a, b, M, log=True)
    u, v = info["u"], info["v"]
    slack = M - (u[:, None] + v[None, :])
    assert slack.min() > -1e-9


def test_dense_complementary_slackness():
    a, b, M = _problem(7, 9)
    G, info = sparse_ot.emd(a, b, M, log=True)
    u, v = info["u"], info["v"]
    mask = G > 1e-12
    residual = (M - u[:, None] - v[None, :])[mask]
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_dense_center_dual_zero_mean():
    a, b, M = _problem(7, 9)
    G, info = sparse_ot.emd(a, b, M, log=True, center_dual=True)
    assert abs(float(info["u"].mean())) < 1e-9


def test_dense_center_dual_preserves_sum():
    a, b, M = _problem(7, 9)
    _, info_t = sparse_ot.emd(a, b, M, log=True, center_dual=True)
    _, info_f = sparse_ot.emd(a, b, M, log=True, center_dual=False)
    s_t = info_t["u"][:, None] + info_t["v"][None, :]
    s_f = info_f["u"][:, None] + info_f["v"][None, :]
    np.testing.assert_allclose(s_t, s_f, atol=1e-9)


def test_sparse_log_dict_and_duality():
    # Banded support (k=7 over n=16) keeps the CSR genuinely sparse — about
    # 41% dense, on the sparse path (avoids the >50%-dense RuntimeWarning
    # under strict CI). Marginals are built from random weights placed on
    # the band itself, guaranteeing primal feasibility.
    rng = np.random.default_rng(3)
    n, m = 16, 16
    k = 7
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(m, lo + k)
        lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            costs.append(rng.uniform(0.1, 1.0))
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    costs = np.asarray(costs, dtype=np.float64)
    w = np.exp(rng.standard_normal(costs.size))
    w /= w.sum()
    a = np.zeros(n)
    b = np.zeros(m)
    np.add.at(a, rows, w)
    np.add.at(b, cols, w)
    a /= a.sum()
    b /= b.sum()
    M_d = np.zeros((n, m))
    M_d[rows, cols] = costs
    M_sp = scipy.sparse.csr_matrix((costs, (rows, cols)), shape=(n, m))
    assert M_sp.nnz / (n * m) < 0.5  # confirm sparse path

    G, info = sparse_ot.emd(a, b, M_sp, log=True)
    assert scipy.sparse.issparse(G)
    assert set(info.keys()) >= {"cost", "u", "v", "warning", "result_code"}

    primal = float(G.multiply(M_d).sum())
    dual = float(a @ info["u"] + b @ info["v"])
    assert abs(primal - dual) < 1e-9
