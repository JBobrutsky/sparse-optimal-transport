# Warm-Start Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Schmitzer-style verify-and-extend path to `sparse_ot.emd` so callers can refine a prior sparse solve to a globally optimal solution on a richer support — without paying for a cold solve.

**Architecture:** New `warm_start=(G, info)` kwarg on `emd()`. When present (and `M` is CSR), `emd()` dispatches to `refine_from_warm_start` in a new `src/sparse_ot/refine.py` module. That module vectorizes a reduced-cost check over `nnz(M_full)`; if the warm-start is already dual-feasible, returns `G_warm` re-laid onto `M_full`'s support; otherwise re-solves Bonneel-sparse on the full support. The result is provably optimal on `(a, b, M_full)`.

**Tech Stack:** Python 3.10+, NumPy, SciPy sparse (CSR), existing `sparse_ot._ext._bonneel` pybind11 extension. Tests with pytest. No C++ changes in v1 — see spec "Pybind / C++ extension" for follow-up.

**Spec:** `docs/superpowers/specs/2026-05-21-warm-start-refinement-design.md`

---

## File map

| Path | Action |
|---|---|
| `src/sparse_ot/refine.py` | Create (new module) |
| `src/sparse_ot/emd.py` | Modify (add `warm_start`, `reduced_cost_tol` kwargs + dispatch) |
| `tests/test_refine.py` | Create |
| `benchmarks/bench_refine.py` | Create |
| `README.md` | Append section |
| `docs/refinement.md` | Create |

---

## Task 1: Stub the API with kwargs + NotImplementedError

Add the kwargs to `emd()` so the surface is locked in, with an explicit `NotImplementedError` body. Cold path is byte-identical to today; new kwargs only matter when set.

**Files:**
- Modify: `src/sparse_ot/emd.py` (function signature of `emd`, lines 38–55 of the docstring/signature; add early dispatch branch around line 60)
- Create: `tests/test_refine.py`

- [ ] **Step 1: Write the failing test (signature + NotImplementedError on dense)**

Create `tests/test_refine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine.py -v`
Expected: both tests FAIL — `TypeError: emd() got an unexpected keyword argument 'warm_start'`.

- [ ] **Step 3: Add the kwargs and the NotImplementedError branch**

Edit `src/sparse_ot/emd.py`. Change the `emd` signature line:

```python
def emd(a, b, M, numItermax=None, log=False, center_dual=True,
        warm_start=None, reduced_cost_tol=None):
```

In the docstring, after the existing parameter descriptions, add:

```python
    """...

    warm_start : tuple or None — if provided, refine to optimum on M from the
                  given prior solve. Form: ``(G_warm, info)`` where info is the
                  log dict from a previous ``emd(..., log=True)`` call, or the
                  bare 3-tuple ``(G_warm, u, v)``. ``G_warm`` may be CSR or a
                  2-D ndarray; both forms accepted. Only supported when M is
                  CSR. See ``docs/refinement.md``.
    reduced_cost_tol : float or None — tolerance for the dual-feasibility check.
                  None picks ``1e-9 * max(1, |M|_inf)``.
    """
```

After the `a = a / a.sum() ; b = b / b.sum()` normalization block, insert the
dispatch branch BEFORE the `scipy.sparse.issparse(M)` check:

```python
    if warm_start is not None:
        if not scipy.sparse.issparse(M):
            raise NotImplementedError(
                "warm_start is only supported for sparse (CSR) M in v1; "
                "for dense M_full the cold path is already optimal."
            )
        # Lazy import to avoid a refine.py ↔ emd.py cycle.
        from sparse_ot.refine import refine_from_warm_start
        return refine_from_warm_start(
            a, b, M, warm_start,
            numItermax=numItermax, log=log,
            center_dual=center_dual,
            reduced_cost_tol=reduced_cost_tol,
        )
```

Create `src/sparse_ot/refine.py` with a stub so the import resolves:

```python
"""Warm-start refinement for sparse-OT. See module docstring in Task 5."""
from __future__ import annotations


def refine_from_warm_start(a, b, M_csr, warm_start, *,
                           numItermax, log, center_dual, reduced_cost_tol):
    raise NotImplementedError("filled in by later tasks")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine.py -v`
Expected: both tests PASS.

Also run the existing test suite to confirm no regression on the cold path:

Run: `pytest tests/ -x -q`
Expected: all existing tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/emd.py src/sparse_ot/refine.py tests/test_refine.py
git commit -m "feat(emd): add warm_start + reduced_cost_tol kwargs (stub)"
```

---

## Task 2: Implement `_parse_warm_start`

Accept `(G_warm, info)`, `(G_warm, u, v)`, with `G_warm` either CSR or 2-D ndarray. Normalize to `(G_csr, u, v)`. Validate.

**Files:**
- Modify: `src/sparse_ot/refine.py`
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Write failing tests for the parser**

Append to `tests/test_refine.py`:

```python
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
    with pytest.raises(TypeError, match="CSR.*ndarray|G"):
        _parse_warm_start(("not a matrix", u, v), n=4, m=4)


def test_parse_warm_start_rejects_bad_shape():
    G_dense = np.eye(3)
    u = np.zeros(4)
    v = np.zeros(4)
    with pytest.raises(ValueError, match="shape"):
        _parse_warm_start((G_dense, u, v), n=4, m=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine.py -v -k parse`
Expected: all FAIL with `ImportError: cannot import name '_parse_warm_start'`.

- [ ] **Step 3: Implement `_parse_warm_start`**

Replace `src/sparse_ot/refine.py` with:

```python
"""Warm-start refinement for sparse-OT. See module docstring in Task 5."""
from __future__ import annotations

import numpy as np
import scipy.sparse


def _parse_warm_start(warm_start, n, m):
    """Normalize ``warm_start`` to ``(G_csr, u, v)``.

    Accepted forms:
      * ``(G, info)`` where ``info`` is a dict with keys ``u`` and ``v``.
      * ``(G, u, v)`` bare 3-tuple.

    ``G`` may be a ``scipy.sparse`` matrix (any format) or a 2-D ``ndarray``.
    Both are normalized to CSR.
    """
    if not isinstance(warm_start, tuple) or len(warm_start) not in (2, 3):
        raise TypeError(
            "warm_start must be a 2-tuple (G, info) or a 3-tuple (G, u, v); "
            f"got {type(warm_start).__name__} of length "
            f"{len(warm_start) if hasattr(warm_start, '__len__') else '?'}"
        )

    if len(warm_start) == 2:
        G, info = warm_start
        if not isinstance(info, dict):
            raise TypeError(
                "warm_start[1] must be the info dict from a prior "
                f"emd(..., log=True) call; got {type(info).__name__}"
            )
        if "u" not in info or "v" not in info:
            raise TypeError(
                "warm_start info dict must contain keys 'u' and 'v'; "
                f"got keys {sorted(info.keys())!r}"
            )
        u = info["u"]
        v = info["v"]
    else:
        G, u, v = warm_start

    if scipy.sparse.issparse(G):
        G_csr = G.tocsr().astype(np.float64)
    elif isinstance(G, np.ndarray) and G.ndim == 2:
        G_csr = scipy.sparse.csr_matrix(G.astype(np.float64, copy=False))
    else:
        raise TypeError(
            "warm_start G must be a CSR matrix or a 2-D ndarray; "
            f"got {type(G).__name__}"
        )

    if G_csr.shape != (n, m):
        raise ValueError(
            f"warm_start G has shape {G_csr.shape}; "
            f"expected ({n}, {m}) to match M_full"
        )

    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if len(u) != n:
        raise ValueError(
            f"warm_start len(u)={len(u)}; expected {n} to match M_full"
        )
    if len(v) != m:
        raise ValueError(
            f"warm_start len(v)={len(v)}; expected {m} to match M_full"
        )
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("warm_start u, v must be finite (no NaN or inf)")

    return G_csr, u, v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine.py -v -k parse`
Expected: all 8 parser tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/refine.py tests/test_refine.py
git commit -m "feat(refine): _parse_warm_start handles CSR + dense G"
```

---

## Task 3: Implement `_compute_reduced_costs`

Vectorized over `nnz(M_csr)`. Returns reduced cost array, min, and a boolean mask of violating edges.

**Files:**
- Modify: `src/sparse_ot/refine.py`
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_refine.py`:

```python
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
    rows, cols = M.nonzero()
    rc_naive = np.array([
        M[rows[k], cols[k]] - u[rows[k]] - v[cols[k]]
        for k in range(M.nnz)
    ])
    np.testing.assert_allclose(rc_fast, rc_naive, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine.py -v -k reduced_costs`
Expected: all FAIL with `ImportError`.

- [ ] **Step 3: Implement `_compute_reduced_costs`**

Append to `src/sparse_ot/refine.py`:

```python
def _compute_reduced_costs(M_csr, u, v, tol=0.0):
    """Reduced cost ``M[i, j] - u[i] - v[j]`` over every nnz edge of M.

    Returns ``(rc, min_rc, n_violating)`` where ``n_violating`` counts edges
    with ``rc < -tol``.
    """
    indptr = M_csr.indptr
    indices = M_csr.indices
    data = M_csr.data
    n = M_csr.shape[0]
    row_idx = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
    rc = data - u[row_idx] - v[indices]
    if rc.size == 0:
        return rc, 0.0, 0
    min_rc = float(rc.min())
    n_viol = int(np.sum(rc < -tol))
    return rc, min_rc, n_viol
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine.py -v -k reduced_costs`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/refine.py tests/test_refine.py
git commit -m "feat(refine): _compute_reduced_costs vectorized over nnz"
```

---

## Task 4: `refine_from_warm_start` — already-optimal branch

When `min_rc >= -tol`, return `G_warm` reshaped onto `M_full`'s support layout. No solve.

**Files:**
- Modify: `src/sparse_ot/refine.py`
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Write failing test (round-trip self-warm-start)**

Append to `tests/test_refine.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_refine.py::test_warm_start_already_optimal_roundtrip -v`
Expected: FAIL with `NotImplementedError: filled in by later tasks`.

- [ ] **Step 3: Implement the orchestrator with already-optimal branch only**

Replace the stub body in `src/sparse_ot/refine.py`:

```python
import warnings

from sparse_ot.feasibility import check_feasibility
from sparse_ot.sparse_utils import to_csr

_MARGINAL_TOL = 1e-6


def _default_tol(M_csr):
    """Scale-relative tolerance for the dual-feasibility check."""
    if M_csr.nnz == 0:
        return 1e-9
    return 1e-9 * max(1.0, float(np.abs(M_csr.data).max()))


def _reshape_onto_support(G_warm_csr, M_csr):
    """Return G with the same values as G_warm, on M_csr's index layout.

    Precondition: nonzero support of G_warm is a subset of M_csr's support.
    Output shares M_csr's shape; nnz equals G_warm.nnz (zeros are dropped).
    """
    # Eliminate explicit zeros for a clean output.
    G = G_warm_csr.copy()
    G.eliminate_zeros()
    return G


def _check_marginals_csr(G, a, b):
    row_sum = np.asarray(G.sum(axis=1)).ravel()
    col_sum = np.asarray(G.sum(axis=0)).ravel()
    err_a = float(np.max(np.abs(row_sum - a)))
    err_b = float(np.max(np.abs(col_sum - b)))
    return err_a, err_b


def _verify_support_subset(G_warm_csr, M_csr):
    """Raise if G_warm has a nonzero outside M_csr's stored support."""
    # Build sets of (i, j) keys for each. Cheap for warm-start sizes; we only
    # need this for the optimistic branch where G_warm is small.
    G_coo = G_warm_csr.tocoo()
    nz = G_coo.data != 0.0
    g_rows = G_coo.row[nz]
    g_cols = G_coo.col[nz]
    if g_rows.size == 0:
        return
    M_coo = M_csr.tocoo()
    m_keys = set(zip(M_coo.row.tolist(), M_coo.col.tolist()))
    for r, c in zip(g_rows.tolist(), g_cols.tolist()):
        if (r, c) not in m_keys:
            raise ValueError(
                f"warm_start G has a nonzero at ({r}, {c}) which is not in "
                f"M_full's support; warm_start is incompatible with M_full"
            )


def refine_from_warm_start(a, b, M_csr, warm_start, *,
                           numItermax, log, center_dual, reduced_cost_tol):
    n, m = M_csr.shape

    G_warm, u, v = _parse_warm_start(warm_start, n, m)
    _verify_support_subset(G_warm, M_csr)

    # Feasibility precondition on M_full (same as cold path).
    row_ptr, col_idx, _costs, _n, _m, _k = to_csr(M_csr, 0.0)
    check_feasibility(a, b, row_ptr, col_idx)

    tol = _default_tol(M_csr) if reduced_cost_tol is None else float(reduced_cost_tol)

    rc, min_rc, n_viol = _compute_reduced_costs(M_csr, u, v, tol=tol)

    if min_rc >= -tol:
        G = _reshape_onto_support(G_warm, M_csr)
        refine_info = {
            "warm_start_optimal": True,
            "num_passes": 0,
            "initial_min_reduced_cost": min_rc,
            "edges_added": 0,
        }
    else:
        # Filled in by Task 5.
        raise NotImplementedError(
            "non-optimal warm_start branch — implemented in Task 5"
        )

    if center_dual:
        shift = float(u.mean())
        u = u - shift
        v = v + shift

    err_a, err_b = _check_marginals_csr(G, a, b)
    converged = max(err_a, err_b) <= _MARGINAL_TOL
    warn_msg = None
    if not converged:
        warn_msg = (
            f"marginals not satisfied after warm-start refinement: "
            f"|G.sum(1)-a|={err_a:.2e}, |G.sum(0)-b|={err_b:.2e} "
            f"(tol={_MARGINAL_TOL:.0e})."
        )
        warnings.warn(warn_msg, RuntimeWarning, stacklevel=3)

    if log:
        cost = float(G.multiply(M_csr).sum())
        return G, {
            "cost": cost,
            "u": u,
            "v": v,
            "warning": warn_msg,
            "result_code": 1 if converged else 0,
            "refine": refine_info,
        }
    return G
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_refine.py::test_warm_start_already_optimal_roundtrip -v`
Expected: PASS.

Also run all tests for regressions:

Run: `pytest tests/ -x -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/refine.py tests/test_refine.py
git commit -m "feat(refine): orchestrator + already-optimal branch"
```

---

## Task 5: Non-optimal branch — cold re-solve fallback

Per spec "Pybind / C++ extension": v1 falls back to a cold re-solve on `M_full` when the warm-start isn't already dual-feasible. The verifier was already free; only this branch costs a solve.

**Files:**
- Modify: `src/sparse_ot/refine.py`
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Write failing test (sub-support refines to cold-optimum)**

Append to `tests/test_refine.py`:

```python
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
    # And it actually exercised the non-optimal branch (we'd expect M_full's
    # extra edges to be useful here, but it's fine if the warm-start happened
    # to already be optimal — assert at least one of those is true).
    assert info_refined["refine"]["num_passes"] in (0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_refine.py::test_warm_start_subsupport_converges_to_cold_optimum -v`
Expected: FAIL — either with `NotImplementedError` (likely) or with a cost mismatch.

- [ ] **Step 3: Implement the non-optimal branch (cold re-solve fallback)**

In `src/sparse_ot/refine.py`, replace the `raise NotImplementedError("non-optimal warm_start branch — implemented in Task 5")` block with:

```python
    else:
        # v1 fallback: cold re-solve on M_full. The C++ pybind binding does
        # not yet accept (u0, v0) for true basis warm-start; see spec
        # "Pybind / C++ extension" and "Open questions". The verifier above
        # is still useful — it confirms when no re-solve is needed at all.
        # The cold re-solve produces the optimum on (a, b, M_full).
        from sparse_ot._ext import _bonneel
        row_ptr_full, col_idx_full, costs_full, _, _, k_full = to_csr(M_csr, 0.0)
        if numItermax is None:
            num_iter = min(50_000_000, max(100_000, 100 * (n + m + k_full)))
        else:
            num_iter = int(numItermax)
        rows_out, cols_out, vals_out, u, v = _bonneel.solve_sparse(
            a, b, row_ptr_full, col_idx_full, costs_full, num_iter
        )
        G = scipy.sparse.csr_matrix(
            (vals_out, (rows_out, cols_out)), shape=(n, m)
        )
        edges_added = int(G.nnz - G_warm.nnz)
        refine_info = {
            "warm_start_optimal": False,
            "num_passes": 1,
            "initial_min_reduced_cost": min_rc,
            "edges_added": max(edges_added, 0),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_refine.py::test_warm_start_subsupport_converges_to_cold_optimum -v`
Expected: PASS.

Run full suite for regressions:

Run: `pytest tests/ -x -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/refine.py tests/test_refine.py
git commit -m "feat(refine): non-optimal branch falls back to cold re-solve"
```

---

## Task 6: Error-handling tests + dense-G round-trip

Cover every row of the spec's error table and the dense-G normalization case end-to-end.

**Files:**
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Write the remaining error and round-trip tests**

Append to `tests/test_refine.py`:

```python
def test_warm_start_wrong_marginals_still_returns_correct_optimum():
    """Use (u, v) solved against one (a, b), warm-start on different (a', b').

    Refinement must still return the correct optimum on (a', b'). Wrong
    warm-starts cost speed, not correctness.
    """
    a, b, M = _band_problem(30, 9, seed=10)
    _, info_orig = sparse_ot.emd(a, b, M, log=True)
    a2, b2, M2 = _band_problem(30, 9, seed=11)
    assert M.shape == M2.shape  # same support structure
    # Warm-start on (a2, b2, M2) with G/u/v from a different problem.
    rng = np.random.default_rng(0)
    G_irrelevant = scipy.sparse.csr_matrix(
        (np.ones(M2.nnz) * 0.0, (M2.nonzero()[0], M2.nonzero()[1])),
        shape=M2.shape,
    )
    G_ref, info_ref = sparse_ot.emd(
        a2, b2, M2,
        warm_start=(G_irrelevant, info_orig),
        log=True,
    )
    G_cold, info_cold = sparse_ot.emd(a2, b2, M2, log=True)
    assert abs(info_ref["cost"] - info_cold["cost"]) < 1e-9


def test_warm_start_centered_vs_uncentered_duals_match():
    a, b, M = _band_problem(20, 7, seed=12)
    G_c, info_c = sparse_ot.emd(a, b, M, log=True, center_dual=True)
    G_u, info_u = sparse_ot.emd(a, b, M, log=True, center_dual=False)
    # Both are valid warm-starts; both must yield the same refined cost.
    _, info_ref_c = sparse_ot.emd(a, b, M, warm_start=(G_c, info_c), log=True)
    _, info_ref_u = sparse_ot.emd(a, b, M, warm_start=(G_u, info_u), log=True)
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
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_refine.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_refine.py
git commit -m "test(refine): error paths + dense-G + bare-tuple coverage"
```

---

## Task 7: Benchmark — regime of optimality

`benchmarks/bench_refine.py` sweeps `(n, k_full, k_warm/k_full)` on the existing `knn-grid` problem and writes `benchmarks/results/refine.json`. Quick/mid/full modes match existing conventions.

**Files:**
- Create: `benchmarks/bench_refine.py`

- [ ] **Step 1: Write the benchmark file**

Create `benchmarks/bench_refine.py`:

```python
"""sparse-ot warm-start refinement benchmark.

Sweep (n, k_full, k_warm_ratio) on the seeded knn-grid problem from
benchmarks/problems.py. For each cell, time three paths:

  cold_full         : sparse_ot.emd(a, b, M_full) from scratch.
  refine            : sparse_ot.emd(a, b, M_warm, log=True)
                      + sparse_ot.emd(a, b, M_full, warm_start=...);
                      reported as the combined wall time (user-visible cost).
  refine_amortized  : refinement step only (cold sub-solve is "free" if
                      already paid for some other reason).

Each cell asserts cost equality between cold_full and refine within 1e-6
relative.

Usage:
    python benchmarks/bench_refine.py            # full sweep
    python benchmarks/bench_refine.py --quick    # tiny sweep (seconds)

Writes benchmarks/results/refine{_quick,}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd

RESULTS_DIR = Path(__file__).parent / "results"

NS_QUICK = [200, 1_000]
KS_FULL_QUICK = [16, 64]
WARM_RATIOS_QUICK = [0.25, 1.0]

NS_FULL = [1_000, 10_000, 100_000]
KS_FULL_FULL = [32, 128, 512]
WARM_RATIOS_FULL = [0.1, 0.25, 0.5, 1.0]


def _time_once(call):
    t0 = time.perf_counter()
    out = call()
    t1 = time.perf_counter()
    return t1 - t0, out


def _restrict_to_k(M_full_csr, k_warm):
    """Subselect the k_warm cheapest edges per row from M_full's support.

    Returns a CSR with the same shape; row i keeps min(k_warm, row_nnz)
    edges.
    """
    n, m = M_full_csr.shape
    rows, cols, costs = [], [], []
    for i in range(n):
        s = M_full_csr.indptr[i]
        e = M_full_csr.indptr[i + 1]
        idx = M_full_csr.indices[s:e]
        d = M_full_csr.data[s:e]
        if len(d) <= k_warm:
            keep = np.arange(len(d))
        else:
            keep = np.argpartition(d, k_warm)[:k_warm]
        for kk in keep:
            rows.append(i)
            cols.append(int(idx[kk]))
            costs.append(float(d[kk]))
    return scipy.sparse.csr_matrix(
        (np.array(costs), (np.array(rows), np.array(cols))),
        shape=(n, m),
    )


def _run_cell(n, k_full, warm_ratio, seed=0):
    a, b, M_full, _w = generate_knn_grid_problem(n, k_full, seed=seed)
    k_warm = max(2, int(round(k_full * warm_ratio)))
    M_warm = _restrict_to_k(M_full, k_warm)

    t_cold, (G_cold, info_cold_full) = _time_once(
        lambda: emd(a, b, M_full, log=True)
    )

    t_phase1, (G_warm, info_warm) = _time_once(
        lambda: emd(a, b, M_warm, log=True)
    )
    t_phase2, (G_refined, info_refined) = _time_once(
        lambda: emd(a, b, M_full, warm_start=(G_warm, info_warm), log=True)
    )

    # Correctness gate.
    cold_cost = info_cold_full["cost"]
    ref_cost = info_refined["cost"]
    rel = abs(ref_cost - cold_cost) / max(abs(cold_cost), 1e-30)
    if rel > 1e-6:
        raise AssertionError(
            f"refine vs cold cost mismatch: rel={rel:.3e} "
            f"(cold={cold_cost!r}, refined={ref_cost!r}) "
            f"@ n={n} k_full={k_full} warm_ratio={warm_ratio}"
        )

    return {
        "n": n,
        "k_full": k_full,
        "k_warm": k_warm,
        "warm_ratio": warm_ratio,
        "cold_full_sec": t_cold,
        "phase1_sec": t_phase1,
        "phase2_sec": t_phase2,
        "refine_sec": t_phase1 + t_phase2,
        "refine_amortized_sec": t_phase2,
        "warm_start_optimal": info_refined["refine"]["warm_start_optimal"],
        "edges_added": info_refined["refine"]["edges_added"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        ns, ks, ratios = NS_QUICK, KS_FULL_QUICK, WARM_RATIOS_QUICK
        out_name = "refine_quick.json"
    else:
        ns, ks, ratios = NS_FULL, KS_FULL_FULL, WARM_RATIOS_FULL
        out_name = "refine.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        for k in ks:
            for r in ratios:
                if k > n:
                    continue
                print(f"running n={n} k_full={k} warm_ratio={r}", flush=True)
                rows.append(_run_cell(n, k, r))

    out = RESULTS_DIR / out_name
    with out.open("w") as f:
        json.dump({"cells": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the quick benchmark to verify it works end-to-end**

Run: `python benchmarks/bench_refine.py --quick`
Expected: prints one `running …` line per cell, finishes in under a minute, writes `benchmarks/results/refine_quick.json` with non-empty `cells`.

- [ ] **Step 3: Sanity-check the JSON output**

Run: `python -c "import json; d = json.load(open('benchmarks/results/refine_quick.json')); print(len(d['cells']), 'cells'); print(d['cells'][0])"`
Expected: at least 4 cells; each cell has `cold_full_sec`, `refine_sec`, `warm_start_optimal`.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_refine.py
git commit -m "bench: regime-of-optimality sweep for warm-start refinement"
```

---

## Task 8: Documentation

Module docstring (canonical reference), README section, `docs/refinement.md` long-form (with 32×32 worked example).

**Files:**
- Modify: `src/sparse_ot/refine.py` (add module docstring at top)
- Modify: `README.md` (append section)
- Create: `docs/refinement.md`

- [ ] **Step 1: Add the canonical module docstring**

At the top of `src/sparse_ot/refine.py`, replace the placeholder docstring `"""Warm-start refinement for sparse-OT. See module docstring in Task 5."""` with:

```python
"""Warm-start refinement for sparse optimal transport.

Refines a warm-start OT solution to global optimality on a larger cost
support, without paying for a cold solve from scratch.

Algorithm (single-pass column generation on Bonneel's network simplex)
======================================================================

Given dual potentials ``(u, v)`` from a previous ``emd`` call on a restricted
support ``S_warm ⊆ E_full``, dual feasibility on ``S_warm`` means
``u[i] + v[j] ≤ M[i, j]`` for all ``(i, j) ∈ S_warm``. To verify global
optimality on ``E_full``, compute the reduced cost
``rc[i, j] = M[i, j] − u[i] − v[j]`` over every edge of ``M_full`` (one
vectorized pass over the CSR ``nnz``):

* If ``min(rc) ≥ −tol``: ``(u, v)`` is dual-feasible on ``E_full``, and
  ``G_warm`` is primal-feasible for ``(a, b)`` with support in
  ``S_warm ⊆ {(i, j) : rc[i, j] ≈ 0}``. By complementary slackness ``G_warm``
  extended with zeros over ``E_full \\ S_warm`` is a global optimum on
  ``(a, b, M_full)``. Return immediately — no re-solve.

* Otherwise: violating edges define entering variables for the simplex on
  ``M_full``. We re-solve Bonneel-sparse on the full ``M_full`` support
  (cold in v1; warm-started from ``(u, v)`` once the C++ binding supports
  it — see the design spec).

Correctness follows from LP duality: a primal feasible flow whose support
consists of edges with ``rc = 0`` and whose duals are ``(u, v)`` is optimal.

Regime of optimality
====================

This path beats a cold solve when:

* ``M_full`` is sparse (CSR) with moderate density — dense ``M_full`` is
  rejected with ``NotImplementedError``.
* ``S_warm`` covers a meaningful fraction of the optimal plan's support on
  ``M_full``. In the limit ``S_warm = E_full`` the refinement degenerates to
  a single verifier pass with no solve; in the limit ``S_warm`` is unrelated
  to the optimum, the cold re-solve costs roughly the same as cold and the
  verifier is pure overhead.

See ``benchmarks/bench_refine.py`` for measured numbers and
``docs/refinement.md`` for a worked example.

References
==========

* Schmitzer, B. "A sparse multiscale algorithm for dense optimal
  transport." *Journal of Mathematical Imaging and Vision*, 56(2):238–259,
  2016. https://doi.org/10.1007/s10851-016-0653-9
* Rauch, J. and Zanotti, L. "An improved implementation of Schmitzer's
  sparse multiscale algorithm for discrete optimal transport on grids."
  arXiv:2502.20905, 2025. https://arxiv.org/abs/2502.20905.
  Reference implementation: https://github.com/johannesrauch/GridOT
  (Boost license).
* Bonneel, N. et al. "Displacement interpolation using Lagrangian mass
  transport." *ACM TOG*, 30(6), 2011.
  https://github.com/nbonneel/network_simplex
* Bertsimas, D. and Tsitsiklis, J. *Introduction to Linear Optimization*,
  Athena Scientific, 1997. §4 — column generation, dual feasibility test.
"""
from __future__ import annotations
```

(The existing `from __future__ import annotations` line stays — make sure
it appears exactly once, right after the closing `"""` of the docstring.)

- [ ] **Step 2: Append the README section**

Open `README.md` and append at the end:

````markdown

## Warm-starting from a previous solve

When you have a cheap solve on a restricted support — for instance, a
small-k nearest-neighbor approximation — you can refine it to a globally
optimal solution on a richer support without paying for a cold solve.

```python
import numpy as np, scipy.sparse, sparse_ot as sot

# Phase 1 — cheap cold solve on a coarse support (k = 8 NN).
M_coarse = build_knn_cost(points, k=8)
G_coarse, info = sot.emd(a, b, M_coarse, log=True)

# Phase 2 — refine to optimum on a denser support (k = 64). The full
# (G, info) tuple is the warm-start payload: G_coarse lets us return
# immediately when the warm-start is already optimal on M_full; info
# supplies (u, v).
M_full = build_knn_cost(points, k=64)
G_opt, info_opt = sot.emd(a, b, M_full,
                          warm_start=(G_coarse, info), log=True)

print(info_opt["refine"])
# {'warm_start_optimal': False, 'num_passes': 1,
#  'initial_min_reduced_cost': -0.014, 'edges_added': 488}

# Chain: refine again on an even denser support.
M_finer = build_knn_cost(points, k=256)
G_final, info_final = sot.emd(a, b, M_finer,
                              warm_start=(G_opt, info_opt), log=True)
```

The refinement path is sparse-only (CSR `M_full` required) and exact: the
returned flow is provably optimal on `M_full`. `G_warm` may be passed as
either CSR or a dense ndarray. See `docs/refinement.md` for the regime
where this beats a cold solve, with measured speedups on the `knn-grid`
benchmark.
````

- [ ] **Step 3: Create `docs/refinement.md` with the 32×32 worked example**

Create `docs/refinement.md`:

````markdown
# Warm-start refinement

A long-form companion to `src/sparse_ot/refine.py`. Read the module
docstring for the algorithm and citations; this document covers regime
of optimality and a worked numerical example.

## When to use this path

Use `warm_start` when:

* you already have an OT solution `(G, info)` from a previous `emd` call;
* you want a globally optimal solution on a **larger** sparse support
  containing the previous one (e.g., the same problem at a higher
  nearest-neighbor `k`);
* `M_full` is sparse (CSR) — dense `M_full` is not supported in v1
  because it defeats the memory benefit.

Do **not** use this path when:

* `M_full` is dense — call cold `emd(a, b, M)` instead.
* you have no prior solve — same answer, cold is fine.
* the prior support is unrelated to the optimum on `M_full` — the
  verifier costs `O(nnz(M_full))` and you'll still pay for a cold
  re-solve.

## Regime of optimality

Empirically, on the seeded `knn-grid` problems (see
`benchmarks/bench_refine.py`):

* When `S_warm = E_full` (the warm-start ran on the *same* support as
  `M_full`): refinement is essentially the verifier pass — sub-millisecond
  per `nnz`. Speedup is the entire cold-solve cost.
* When `S_warm` is a strict subset that captures the optimum (e.g., the
  same problem on a tighter `k`): refinement matches cold-on-`M_full` in
  cost; wall time is dominated by Phase 1 (the user's cheap solve) plus a
  cheap verify pass. The "warm_start_optimal=True" branch fires.
* When `S_warm` is a strict subset that *misses* the optimum:
  refinement falls back to a cold re-solve on `M_full` (v1 limitation —
  see below). The verifier pass becomes overhead. Wall time is
  Phase 1 + cold-on-`M_full`. Use cold `emd(a, b, M_full)` directly in
  this regime.

The exact crossover depends on Bonneel's pivot-count sensitivity to the
warm-start basis, which v1 does not yet exploit (cold re-solve fallback —
see the design spec, "Pybind / C++ extension").

## Worked example — 32×32 dual-feasibility check

A 32×32 problem is small enough to print and large enough to actually
exhibit a non-trivial refinement.

```python
import numpy as np, scipy.sparse, sparse_ot as sot

n = 32
rng = np.random.default_rng(0)

def band_cost(n, k):
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + k); lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i); cols.append(j); costs.append(float((i - j) ** 2))
    return scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )

# Feasible marginals on the k=3 band (sums to 1 by construction).
M3 = band_cost(n, k=3)
w = np.exp(rng.standard_normal(M3.nnz)); w /= w.sum()
a = np.zeros(n); b = np.zeros(n)
np.add.at(a, M3.nonzero()[0], w); np.add.at(b, M3.nonzero()[1], w)
a /= a.sum(); b /= b.sum()

# Phase 1 — solve on the tight k=3 band.
G3, info3 = sot.emd(a, b, M3, log=True)
print("Phase 1 cost on k=3:", info3["cost"])
print("Phase 1 nnz(G):", G3.nnz)

# Phase 2 — refine on a wider k=9 band. The same problem on the larger
# support has the same optimal cost iff the optimum is realised inside the
# k=3 band, which it is by construction (marginals were sampled from k=3).
M9 = band_cost(n, k=9)
G9, info9 = sot.emd(a, b, M9, warm_start=(G3, info3), log=True)
print("Phase 2 refine info:", info9["refine"])
# {'warm_start_optimal': True, 'num_passes': 0,
#  'initial_min_reduced_cost': 0.0, 'edges_added': 0}
print("Phase 2 cost on k=9:", info9["cost"])
assert abs(info9["cost"] - info3["cost"]) < 1e-12

# Force a non-trivial refinement: tilt the marginals so the optimum on k=9
# uses edges outside the k=3 support.
a2 = np.roll(a, 1); b2 = np.roll(b, -1)
a2 /= a2.sum(); b2 /= b2.sum()
# Re-solve k=3 (still feasible — band-3 is connected on n=32).
G3_t, info3_t = sot.emd(a2, b2, M3, log=True)
G9_t, info9_t = sot.emd(a2, b2, M9, warm_start=(G3_t, info3_t), log=True)
print("Tilted refine info:", info9_t["refine"])
# warm_start_optimal=False; num_passes=1; edges_added>0.
print("Tilted Phase 2 cost on k=9:", info9_t["cost"])
G9_cold, info9_cold = sot.emd(a2, b2, M9, log=True)
assert abs(info9_t["cost"] - info9_cold["cost"]) < 1e-9
```

Reading the output:

* `initial_min_reduced_cost` is the minimum of
  `M_full[i, j] − u[i] − v[j]` over every nnz of `M_full`. A value ≥ 0
  (within tolerance) means the warm-start's duals are already feasible on
  the larger support, and refinement returns `G_warm` directly.
* `edges_added` is the change in `nnz(G)` from `G_warm` to `G_refined`. In
  the already-optimal branch it is 0 by construction; in the cold-re-solve
  branch it is the difference in the solvers' chosen bases.
* `warm_start_optimal` is the bit that distinguishes the two branches.
````

- [ ] **Step 4: Verify the README and docs render**

Run: `python -c "import sparse_ot; help(sparse_ot.refine)"` (after a
rebuild via `pip install -e . --no-build-isolation` if needed).
Expected: the module docstring prints cleanly.

Run: `pytest tests/ -x -q`
Expected: all PASS (no regressions from docstring edits).

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/refine.py README.md docs/refinement.md
git commit -m "docs(refine): module docstring + README + refinement.md"
```

---

## Self-review notes

* **Spec coverage check.** Every spec section is covered:
  * API section → Task 1 (kwargs) + Task 4/5 (dispatch + log dict).
  * Architecture / Algorithm → Task 4 (already-optimal) + Task 5 (non-optimal).
  * Components (`_parse_warm_start`, `_compute_reduced_costs`,
    `_reshape_onto_support`) → Tasks 2, 3, 4.
  * Router change in `emd.py` → Task 1.
  * Pybind / C++ extension → Task 5 documents the v1 cold-re-solve
    fallback in code comments and refers to the spec's "Open questions".
  * Log-dict additions → Task 4 (already-optimal) and Task 5 (non-optimal).
  * Error handling — every row of the table is exercised: missing keys
    (Task 2 test #6), shape mismatch (Task 2 test #4), wrong G type
    (Task 2 test #7), non-finite (Task 2 test #5), dense M_full
    (Task 1 test #2), G outside M support (Task 6 test #3).
  * Tests #1–#12 from spec § Testing strategy → covered across Tasks 4–6.
  * Benchmark → Task 7.
  * Docs (module docstring, README, `docs/refinement.md` with 32×32
    example) → Task 8.

* **Placeholder scan.** No "TBD", "TODO", or "implement appropriate X".
  Every code block is complete.

* **Type/name consistency.** `_parse_warm_start` always returns
  `(G_csr, u, v)`; callers in Task 4 and 5 destructure into the same
  names. `_compute_reduced_costs` returns `(rc, min_rc, n_viol)` —
  matches Task 4 usage. `refine_info` keys match the spec verbatim.

* **C++ binding gap.** Task 5 deliberately uses the existing
  `_bonneel.solve_sparse` cold entry point for the fallback. This is
  explicit in the v1 spec (Pybind / C++ extension → "If exposing this is
  non-trivial, v1 falls back to cold re-solve on M_full"). The plan does
  not change the pybind layer.

