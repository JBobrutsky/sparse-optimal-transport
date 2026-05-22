import warnings

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


def test_warm_start_with_sparse_M_non_optimal_produces_correct_result():
    """Sparse M + non-optimal warm_start goes through the non-optimal branch
    and produces the correct optimum. warm_start_optimal=False, num_passes=1."""
    import warnings
    n = 10
    a, b, M = _band_problem(n, 5, seed=0)
    G_cold, info_cold = sparse_ot.emd(a, b, M, log=True)
    cold_cost = info_cold["cost"]

    # Build an explicitly sub-optimal warm start:
    # 1-arc G (highly degenerate → Mode C) with u[0]=0.5 to force min_rc < 0.
    # M[0,0]=0 so rc(0,0) = 0 - 0.5 - 0 = -0.5 < -tol → non-optimal branch fires.
    coo = G_cold.tocoo()
    idx = int(np.argmax(coo.data))
    G_deg = scipy.sparse.csr_matrix(
        ([coo.data[idx]], ([coo.row[idx]], [coo.col[idx]])), shape=M.shape
    )
    u_bad = np.zeros(n); u_bad[0] = 0.5
    v_bad = np.zeros(n)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        G_warm, info_warm = sparse_ot.emd(
            a, b, M, warm_start=(G_deg, u_bad, v_bad), log=True
        )
    assert info_warm["refine"]["warm_start_optimal"] is False
    assert info_warm["refine"]["num_passes"] == 1
    assert abs(info_warm["cost"] - cold_cost) < 1e-9
    np.testing.assert_allclose(np.asarray(G_warm.sum(axis=1)).ravel(), a, atol=1e-6)
    np.testing.assert_allclose(np.asarray(G_warm.sum(axis=0)).ravel(), b, atol=1e-6)


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


from sparse_ot.refine import _compute_reduced_costs


def test_reduced_costs_all_nonneg_for_optimal_duals():
    """rc = M - u[i] - v[j] >= 0 for the optimum's dual potentials."""
    a, b, M = _band_problem(20, 9, seed=1)
    G, info = sparse_ot.emd(a, b, M, log=True)
    rc, min_rc, n_viol = _compute_reduced_costs(M, info["u"], info["v"])
    assert rc.shape == (M.nnz,)
    assert min_rc >= -1e-9
    assert n_viol == 0


def test_reduced_costs_negative_when_duals_are_wrong():
    """Zero duals make rc = M, but a hand-tweaked u/v can make some negative."""
    a, b, M = _band_problem(10, 5, seed=2)
    u = np.zeros(10)
    v = np.full(10, M.data.max() + 1.0)  # u+v > M[i,j] for every edge
    rc, min_rc, n_viol = _compute_reduced_costs(M, u, v)
    assert min_rc < 0
    assert n_viol == M.nnz


def test_reduced_costs_matches_dense_formula():
    """Vectorized rc equals the naive (i, j) loop on M.toarray()."""
    a, b, M = _band_problem(12, 5, seed=3)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(12)
    v = rng.standard_normal(12)
    rc_fast, _, _ = _compute_reduced_costs(M, u, v)
    # Iterate via CSR structure so explicit zero entries are included,
    # matching the vectorized implementation's nnz traversal.
    rc_naive = np.empty(M.nnz)
    for i in range(M.shape[0]):
        for ptr in range(M.indptr[i], M.indptr[i + 1]):
            j = M.indices[ptr]
            rc_naive[ptr] = M.data[ptr] - u[i] - v[j]
    np.testing.assert_allclose(rc_fast, rc_naive, atol=1e-12)


def test_warm_start_already_optimal_roundtrip():
    """Solve cold, feed (G, info) back as warm_start on the same problem.

    Expected: warm_start_optimal=True, num_passes=0, identical costs and
    plans within 1e-12.
    """
    a, b, M = _band_problem(30, 7, seed=4)
    G_cold, info_cold = sparse_ot.emd(a, b, M, log=True)

    G_refined, info_refined = sparse_ot.emd(
        a, b, M, warm_start=(G_cold, info_cold), log=True
    )

    assert info_refined["refine"]["warm_start_optimal"] is True
    assert info_refined["refine"]["num_passes"] == 0
    assert info_refined["refine"]["edges_added"] == 0
    assert abs(info_refined["cost"] - info_cold["cost"]) < 1e-12
    np.testing.assert_allclose(
        G_refined.toarray(), G_cold.toarray(), atol=1e-12
    )


def _build_band_M(n, k):
    """Build a (n, n) k-NN band CSR with costs (i - j)^2. No randomness."""
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
    return scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )


def test_warm_start_subsupport_converges_to_cold_optimum():
    """Warm-start from a k=5 band; refine on k=15 band; cost matches a
    fresh cold solve on the k=15 band within 1e-9."""
    n = 50
    rng = np.random.default_rng(7)
    M_warm = _build_band_M(n, 5)
    M_full = _build_band_M(n, 15)

    # Marginals that are feasible on the WARM support (so phase 1 converges).
    half = 5 // 2
    rows = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + 5)
        lo = max(0, hi - 5)
        rows.extend([i] * (hi - lo))
    w = np.exp(rng.standard_normal(len(rows)))
    w /= w.sum()
    a = np.zeros(n)
    b = np.zeros(n)
    cols = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + 5)
        lo = max(0, hi - 5)
        cols.extend(range(lo, hi))
    rows_arr = np.asarray(rows)
    cols_arr = np.asarray(cols)
    np.add.at(a, rows_arr, w)
    np.add.at(b, cols_arr, w)
    a /= a.sum()
    b /= b.sum()

    G_warm, info_warm = sparse_ot.emd(a, b, M_warm, log=True)
    G_full_cold, info_full_cold = sparse_ot.emd(a, b, M_full, log=True)
    G_refined, info_refined = sparse_ot.emd(
        a, b, M_full, warm_start=(G_warm, info_warm), log=True
    )

    # Refinement matches cold-on-full to within 1e-9 (LP-grade).
    assert abs(info_refined["cost"] - info_full_cold["cost"]) < 1e-9
    np.testing.assert_allclose(
        G_refined.toarray(), G_full_cold.toarray(), atol=1e-9
    )
    # The refinement either found the warm-start already optimal (0 passes)
    # or needed exactly one re-solve.
    assert info_refined["refine"]["num_passes"] in (0, 1)


def test_warm_start_suboptimal_duals_triggers_fallback():
    """Warm-start with infeasible duals triggers exactly one cold re-solve.

    Checks: warm_start_optimal=False, num_passes=1, initial_min_reduced_cost<0,
    and the resulting plan satisfies marginals.  This is distinct from
    test_warm_start_with_sparse_M_non_optimal_uses_cold_resolve in that it also
    validates the reported initial_min_reduced_cost is genuinely negative.
    """
    a, b, M = _band_problem(20, 5, seed=10)
    # Create a deliberately bad warm-start: empty flow with hand-made duals.
    G_bad = scipy.sparse.csr_matrix(M.shape, dtype=np.float64)
    u_bad = np.zeros(20)
    # large offset guarantees rc = M - u - v < 0 on every stored edge
    v_bad = np.full(20, M.data.max() + 10.0)
    # Solve with the bad warm-start. The empty G_bad has 100% zero-flow basic
    # arcs, which intentionally triggers the degenerate-basis RuntimeWarning;
    # that warning is the expected path here (forces potential-only fallback).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        G, info = sparse_ot.emd(
            a, b, M,
            warm_start=(G_bad, u_bad, v_bad),
            log=True,
        )
    assert info["refine"]["warm_start_optimal"] is False
    assert info["refine"]["num_passes"] == 1
    # The initial reduced costs must have been negative (that is what forced the re-solve).
    assert info["refine"]["initial_min_reduced_cost"] < 0
    # And it should still converge to the right answer.
    np.testing.assert_allclose(np.asarray(G.sum(axis=1)).ravel(), a, atol=1e-6)
    np.testing.assert_allclose(np.asarray(G.sum(axis=0)).ravel(), b, atol=1e-6)


def test_warm_start_centered_vs_uncentered_duals_match():
    """Centered and uncentered duals both pass the reduced-cost check.

    A cold solve with center_dual=True shifts (u, v) by a constant; that
    shift must not cause the warm-start to be falsely classified as
    suboptimal.  Both variants should be detected as already-optimal
    (warm_start_optimal=True, num_passes=0) and produce the same cost.
    """
    a, b, M = _band_problem(20, 7, seed=12)
    G_c, info_c = sparse_ot.emd(a, b, M, log=True, center_dual=True)
    G_u, info_u = sparse_ot.emd(a, b, M, log=True, center_dual=False)
    # Both are valid warm-starts; both must be detected as already-optimal.
    _, info_ref_c = sparse_ot.emd(a, b, M, warm_start=(G_c, info_c), log=True)
    _, info_ref_u = sparse_ot.emd(a, b, M, warm_start=(G_u, info_u), log=True)
    assert info_ref_c["refine"]["warm_start_optimal"] is True
    assert info_ref_u["refine"]["warm_start_optimal"] is True
    assert info_ref_c["refine"]["num_passes"] == 0
    assert info_ref_u["refine"]["num_passes"] == 0
    assert abs(info_ref_c["cost"] - info_ref_u["cost"]) < 1e-12


def test_warm_start_G_outside_M_support_raises():
    a, b, M = _band_problem(10, 3, seed=13)
    # Build G with a nonzero at (0, n-1), which is far outside the band.
    G_bad = scipy.sparse.csr_matrix(
        (np.array([0.5]), (np.array([0]), np.array([9]))),
        shape=(10, 10),
    )
    u = np.zeros(10)
    v = np.zeros(10)
    with pytest.raises(ValueError, match="not in M_full"):
        sparse_ot.emd(a, b, M, warm_start=(G_bad, u, v))


def test_warm_start_bare_tuple_matches_dict_form():
    """(G, u, v) 3-tuple and (G, info_dict) 2-tuple warm-starts are equivalent.

    Both normalization paths in _parse_warm_start must produce the same result
    when G and the dual potentials are identical.
    """
    a, b, M = _band_problem(20, 7, seed=14)
    G_cold, info_cold = sparse_ot.emd(a, b, M, log=True)
    _, info_dict = sparse_ot.emd(
        a, b, M, warm_start=(G_cold, info_cold), log=True
    )
    _, info_tuple = sparse_ot.emd(
        a, b, M,
        warm_start=(G_cold, info_cold["u"], info_cold["v"]),
        log=True,
    )
    assert info_dict["cost"] == info_tuple["cost"]
    assert info_dict["refine"]["warm_start_optimal"] == \
        info_tuple["refine"]["warm_start_optimal"]


def test_warm_start_dense_G_matches_csr_G():
    """Dense G_warm produces identical refined output to CSR G_warm."""
    a, b, M = _band_problem(20, 7, seed=15)
    G_cold, info_cold = sparse_ot.emd(a, b, M, log=True)
    G_dense = G_cold.toarray()
    _, info_csr = sparse_ot.emd(
        a, b, M, warm_start=(G_cold, info_cold), log=True
    )
    _, info_dense = sparse_ot.emd(
        a, b, M,
        warm_start=(G_dense, info_cold["u"], info_cold["v"]),
        log=True,
    )
    assert info_csr["cost"] == info_dense["cost"]
    assert info_csr["refine"]["warm_start_optimal"] == \
        info_dense["refine"]["warm_start_optimal"]


def test_refine_non_optimal_warm_basis_used():
    """Non-optimal branch: warm_basis_used=True when G_warm is non-degenerate.

    We solve on M6 with perturbed (non-uniform) marginals to get a full-rank
    spanning-tree basis G6, then refine on M10 using deliberately wrong duals
    (zeros) so the reduced-cost check fails and the non-optimal branch fires.
    The G6 plan is still primal-feasible for (a, b), so bonneel_sparse_solve_warm
    receives a non-degenerate warm basis and Mode B should fire.
    """
    n = 40
    half = 3
    rows, cols, costs_list = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + 6)
        lo = max(0, hi - 6)
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            costs_list.append(float((i - j) ** 2) + 0.01)
    M6 = scipy.sparse.csr_matrix((costs_list, (rows, cols)), shape=(n, n))

    half2 = 5
    rows2, cols2, costs2_list = [], [], []
    for i in range(n):
        lo = max(0, i - half2)
        hi = min(n, lo + 10)
        lo = max(0, hi - 10)
        for j in range(lo, hi):
            rows2.append(i)
            cols2.append(j)
            costs2_list.append(float((i - j) ** 2) + 0.01)
    M10 = scipy.sparse.csr_matrix((costs2_list, (rows2, cols2)), shape=(n, n))

    # Perturbed marginals to avoid degenerate BFS (uniform marginals trigger Mode C)
    rng = np.random.default_rng(99)
    a = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    a = np.abs(a)
    a /= a.sum()
    b = np.ones(n) / n + rng.uniform(-0.001, 0.001, n)
    b = np.abs(b)
    b /= b.sum()

    from sparse_ot import emd
    G6, info6 = emd(a, b, M6, log=True)
    G10_cold, info10_cold = emd(a, b, M10, log=True)

    # Use zero duals — guaranteed to violate reduced costs on M10 because
    # rc = M10[i,j] - 0 - 0 = M10[i,j] > 0 always ... except M10 has
    # zero-cost diagonal entries ((i-j)^2+0.01 > 0 always), so all rc >= 0.01 > 0.
    # Instead use u = max(M10)/2 to force negative reduced costs.
    u_bad = np.full(n, float(M10.data.max()) / 2.0)
    v_bad = np.full(n, float(M10.data.max()) / 2.0)

    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        G10_warm, info10_warm = emd(a, b, M10, warm_start=(G6, u_bad, v_bad), log=True)

    assert abs(info10_warm["cost"] - info10_cold["cost"]) < 1e-9
    refine = info10_warm["refine"]
    assert refine["warm_start_optimal"] is False
    assert refine["warm_basis_used"] is True  # non-degenerate G6


def test_refine_non_optimal_mode_c_fallback():
    """Non-optimal branch: warm_basis_used=False when G_warm is degenerate.

    We construct a degenerate G_warm with only 1 arc and pair it with bad
    duals (large offsets that make every reduced cost negative) so the
    reduced-cost check fails and the non-optimal branch fires.  With only 1
    arc in G_warm (far below the 5% degeneracy threshold), Mode C must fire.
    """
    import warnings
    n = 30
    rows, cols, costs_list = [], [], []
    for i in range(n):
        for j in range(max(0, i - 2), min(n, i + 3)):
            rows.append(i)
            cols.append(j)
            costs_list.append(float((i - j) ** 2) + 0.1)
    M = scipy.sparse.csr_matrix((costs_list, (rows, cols)), shape=(n, n))
    a = np.ones(n) / n
    b = np.ones(n) / n

    from sparse_ot import emd
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    # Construct degenerate G_warm: keep only 1 arc (far below 5% of n+m-1)
    coo = G_cold.tocoo()
    idx = np.argmax(coo.data)
    G_deg = scipy.sparse.csr_matrix(
        ([coo.data[idx]], ([coo.row[idx]], [coo.col[idx]])), shape=M.shape
    )

    # Use bad duals that force the reduced-cost check to fail (non-optimal branch).
    # Large u+v makes rc = M[i,j] - u[i] - v[j] < 0 for every edge.
    u_bad = np.full(n, float(M.data.max()) + 1.0)
    v_bad = np.zeros(n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G_warm, info_warm = emd(a, b, M, warm_start=(G_deg, u_bad, v_bad), log=True)
    assert abs(info_warm["cost"] - cold_cost) < 1e-9
    refine = info_warm["refine"]
    assert refine["warm_basis_used"] is False
    assert any("zero-flow basic arcs" in str(x.message) for x in w)
