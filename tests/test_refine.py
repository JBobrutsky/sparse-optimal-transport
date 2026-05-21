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


def test_warm_start_with_sparse_M_reaches_refine_stub():
    """Sparse M + non-None warm_start dispatches into refine.py's stub,
    which raises NotImplementedError until later tasks fill it in.
    """
    a, b, M = _band_problem(10, 3)
    fake_warm = (
        scipy.sparse.csr_matrix(M.shape),
        np.zeros(10),
        np.zeros(10),
    )
    with pytest.raises(NotImplementedError, match="later tasks"):
        sparse_ot.emd(a, b, M, warm_start=fake_warm)


from sparse_ot.refine import _parse_warm_start


def _trivial_warm(n, m):
    G = scipy.sparse.csr_matrix(np.eye(n, m) / min(n, m))
    u = np.zeros(n)
    v = np.zeros(m)
    return G, u, v


def test_parse_warm_start_tuple_form():
    G, u, v = _trivial_warm(4, 4)
    G_out, u_out, v_out = _parse_warm_start((G, u, v), n=4, m=4)
    assert scipy.sparse.issparse(G_out)
    np.testing.assert_array_equal(u_out, u)
    np.testing.assert_array_equal(v_out, v)


def test_parse_warm_start_info_dict_form():
    G, u, v = _trivial_warm(4, 4)
    info = {"u": u, "v": v, "cost": 0.0,
            "warning": None, "result_code": 1}
    G_out, u_out, v_out = _parse_warm_start((G, info), n=4, m=4)
    np.testing.assert_array_equal(u_out, u)
    np.testing.assert_array_equal(v_out, v)


def test_parse_warm_start_dense_G_normalized_to_csr():
    G_dense = np.eye(4) * 0.25
    u = np.zeros(4)
    v = np.zeros(4)
    G_out, _, _ = _parse_warm_start((G_dense, u, v), n=4, m=4)
    assert scipy.sparse.issparse(G_out)
    assert G_out.shape == (4, 4)
    np.testing.assert_allclose(G_out.toarray(), G_dense)


def test_parse_warm_start_rejects_mismatched_u_length():
    G, u, v = _trivial_warm(4, 4)
    with pytest.raises(ValueError, match="len.u."):
        _parse_warm_start((G, np.zeros(3), v), n=4, m=4)


def test_parse_warm_start_rejects_nan_v():
    G, u, v = _trivial_warm(4, 4)
    v_bad = v.copy()
    v_bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _parse_warm_start((G, u, v_bad), n=4, m=4)


def test_parse_warm_start_rejects_missing_keys():
    G, u, v = _trivial_warm(4, 4)
    bad_info = {"cost": 0.0}  # no u, no v
    with pytest.raises(TypeError, match="u"):
        _parse_warm_start((G, bad_info), n=4, m=4)


def test_parse_warm_start_rejects_wrong_G_type():
    u = np.zeros(4)
    v = np.zeros(4)
    with pytest.raises(TypeError, match="CSR matrix or a 2-D ndarray"):
        _parse_warm_start(("not a matrix", u, v), n=4, m=4)


def test_parse_warm_start_rejects_bad_shape():
    G_dense = np.eye(3)
    u = np.zeros(4)
    v = np.zeros(4)
    with pytest.raises(ValueError, match="shape"):
        _parse_warm_start((G_dense, u, v), n=4, m=4)


def test_parse_warm_start_accepts_coo_input():
    """Non-CSR sparse input is normalized to CSR."""
    G_coo = scipy.sparse.coo_matrix(np.eye(4) * 0.25)
    u = np.zeros(4)
    v = np.zeros(4)
    G_out, _, _ = _parse_warm_start((G_coo, u, v), n=4, m=4)
    assert scipy.sparse.isspmatrix_csr(G_out)
    np.testing.assert_allclose(G_out.toarray(), G_coo.toarray())


def test_parse_warm_start_accepts_int_dtype_G():
    """Integer-dtype G is normalized to float64 CSR."""
    G_int = np.eye(4, dtype=np.int64)
    u = np.zeros(4)
    v = np.zeros(4)
    G_out, _, _ = _parse_warm_start((G_int, u, v), n=4, m=4)
    assert G_out.dtype == np.float64
    np.testing.assert_allclose(G_out.toarray(), G_int.astype(np.float64))


def test_parse_warm_start_rejects_2d_u():
    """A 2-D u (e.g., column vector) is rejected, not silently flattened."""
    G, _, _ = _trivial_warm(4, 4)
    u_bad = np.zeros((4, 1))
    v = np.zeros(4)
    with pytest.raises(ValueError, match="1-D"):
        _parse_warm_start((G, u_bad, v), n=4, m=4)
