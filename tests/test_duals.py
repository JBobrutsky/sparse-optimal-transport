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
