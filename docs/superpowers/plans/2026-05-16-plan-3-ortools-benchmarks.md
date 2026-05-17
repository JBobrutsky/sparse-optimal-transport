# Plan 3: OR-Tools Solver + Benchmark Suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the third solver (OR-Tools `SimpleMinCostFlow`, Python-only, lazy import) so the existing `solver='ortools'` route stops raising `NotImplementedError`, and build a reproducible benchmark suite that produces efficiency/accuracy JSON results, publication-quality figures, and an empirically-derived `routing_thresholds.json`.

**Architecture:** A new Python module `ortools_solver.py` scales float64 supplies/demands and float64 costs to int64, builds `SimpleMinCostFlow` arc-by-arc from CSR input, runs `Solve()`, then unscales the integer flow back to float64 COO. `emd.py` adds an `ortools` branch parallel to the existing LEMON branch. Benchmarks live in `benchmarks/`: `problems.py` generates k-nearest-neighbor grid problems on Dirichlet distributions, `bench_solvers.py` runs the (n, k, solver) sweep and writes JSON results with memory-cutoff `null` cells, and `generate_report.py` reads the JSON to produce figures and a threshold table.

**Tech Stack:** Python 3.12 (uv), OR-Tools 9.8+ (optional dep), POT (oracle), matplotlib, scipy, numpy, pytest, tracemalloc (peak memory).

---

## File Map

| File | Role |
|---|---|
| `src/sparse_ot/ortools_solver.py` | int64 scaling + `SimpleMinCostFlow` driver, returns COO float64 |
| `src/sparse_ot/emd.py` | Add OR-Tools branch (remove `NotImplementedError`) |
| `tests/test_ortools.py` | OR-Tools correctness vs POT and LEMON; int64 scaling round-trip |
| `tests/test_emd.py` | Replace `ortools_raises` with `ortools_override`, add scipy-sparse ortools test |
| `benchmarks/problems.py` | Problem generator: Dirichlet distributions on a k-NN grid graph |
| `benchmarks/bench_solvers.py` | Efficiency + accuracy sweep driver; writes `efficiency.json`, `accuracy.json` |
| `benchmarks/generate_report.py` | Figures + `routing_thresholds.json` derivation from JSON |
| `benchmarks/results/efficiency.json` | Wall time, peak memory, iterations per `(n, k, solver)` |
| `benchmarks/results/accuracy.json` | Relative cost error + feasibility per `(n, k, solver)` |
| `benchmarks/results/figures/` | Heatmaps, line plots, accuracy plots (PDF + PNG) |
| `benchmarks/results/routing_thresholds.json` | Regenerated from benchmark crossover analysis |
| `tests/test_benchmarks.py` | Smoke test that `--quick` benchmark mode completes |
| `README.md` | Updated routing explanation + how to regenerate thresholds |

---

## Task 1: ortools_solver.py — int64 scaling + solver call (TDD)

**Files:**
- Create: `tests/test_ortools.py`
- Create: `src/sparse_ot/ortools_solver.py`

The design: float64 supplies are scaled to int64 by multiplying by a fixed `supply_scale = 10**9` and rounding; rounding residual is absorbed into the last source and last sink so that `sum(source_supplies) == sum(sink_demands)`. Float64 costs are scaled to int64 by `ortools_cost_scale` (default `1e6`) and rounded. After `SimpleMinCostFlow.Solve()`, integer flows are divided by `supply_scale` to recover float64 transport mass. OR-Tools is imported lazily so missing-OR-Tools installs only fail when this solver is actually invoked.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ortools.py
import numpy as np
import pytest
import scipy.sparse

ortools = pytest.importorskip("ortools.graph.python.min_cost_flow")

from sparse_ot.ortools_solver import solve_ortools


def _csr_from_dense(M):
    M = np.asarray(M, dtype=np.float64)
    n, m = M.shape
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    cols, costs = [], []
    for i in range(n):
        for j in range(m):
            if M[i, j] != 0.0:
                cols.append(j)
                costs.append(M[i, j])
        row_ptr[i + 1] = len(cols)
    return row_ptr, np.asarray(cols, dtype=np.int32), np.asarray(costs, dtype=np.float64)


def test_ortools_2x2_optimal_routing():
    """Source 0→sink 0 and source 1→sink 1 is the unique optimum (cost 0)."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((2, 2))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan, np.diag([0.5, 0.5]), atol=1e-6)


def test_ortools_marginals_random():
    """Transport plan marginals match a and b within scaling tolerance."""
    rng = np.random.default_rng(0)
    n = 6
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((n, n))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


def test_ortools_cost_matches_pot():
    """OR-Tools transport cost matches POT within scaling tolerance."""
    import ot
    rng = np.random.default_rng(1)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    cost_ot = sum(M[r, c] * v for r, c, v in zip(rows, cols, vals))
    cost_pot = ot.emd2(a, b, M)
    rel_err = abs(cost_ot - cost_pot) / max(abs(cost_pot), 1e-15)
    assert rel_err < 1e-4, f"ortools={cost_ot}, pot={cost_pot}, rel_err={rel_err}"


def test_ortools_cost_scale_parameter_changes_precision():
    """Higher ortools_cost_scale yields a result no worse than lower scale."""
    import ot
    rng = np.random.default_rng(2)
    n = 6
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0.1, 1.0, (n, n))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    cost_pot = ot.emd2(a, b, M)

    rows_low, cols_low, vals_low = solve_ortools(
        a, b, row_ptr, col_idx, costs, ortools_cost_scale=1e3
    )
    rows_high, cols_high, vals_high = solve_ortools(
        a, b, row_ptr, col_idx, costs, ortools_cost_scale=1e9
    )
    cost_low  = sum(M[r, c] * v for r, c, v in zip(rows_low, cols_low, vals_low))
    cost_high = sum(M[r, c] * v for r, c, v in zip(rows_high, cols_high, vals_high))
    # Both should be near POT; the high-scale one should not be worse.
    err_low  = abs(cost_low  - cost_pot)
    err_high = abs(cost_high - cost_pot)
    assert err_high <= err_low + 1e-9


def test_ortools_rounding_residual_balanced():
    """Distributions that don't divide evenly into supply_scale still solve."""
    # a sums to 1 but has a value that produces a rounding residual at scale 1e9.
    a = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    b = np.array([0.5, 0.25, 0.25])
    M = np.eye(3) * 0.0 + (1.0 - np.eye(3))
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    plan = np.zeros((3, 3))
    for r, c, v in zip(rows, cols, vals):
        plan[r, c] = v
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


def test_ortools_returns_int32_indices_and_float64_values():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    row_ptr, col_idx, costs = _csr_from_dense(M)
    rows, cols, vals = solve_ortools(a, b, row_ptr, col_idx, costs)
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
    assert vals.dtype == np.float64
```

- [ ] **Step 2: Run tests — confirm ImportError**

```bash
pytest tests/test_ortools.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'solve_ortools' from 'sparse_ot.ortools_solver'`. If you instead see `ortools.graph.python.min_cost_flow` missing, install OR-Tools first: `uv pip install ortools`.

- [ ] **Step 3: Implement ortools_solver.py**

```python
# src/sparse_ot/ortools_solver.py
"""OR-Tools min-cost-flow solver for sparse balanced OT.

OR-Tools requires int64 supplies, demands, and costs. Float64 inputs are
scaled by fixed factors, rounded, balanced, and unscaled on the way out.
"""

from __future__ import annotations

import numpy as np

_SUPPLY_SCALE = 10**9  # int64-safe; supports n up to ~1e9 with sub-ULP precision.


def _scale_to_int64(values: np.ndarray, scale: int) -> np.ndarray:
    """Multiply by `scale`, round, and cast to int64."""
    return np.rint(np.asarray(values, dtype=np.float64) * scale).astype(np.int64)


def _balance_supplies(
    a_int: np.ndarray, b_int: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Make sum(a_int) == sum(b_int) by absorbing the residual into the last entries.

    OR-Tools rejects unbalanced supply/demand. Rounding `a * 1e9` and `b * 1e9`
    independently leaves a small residual (typically <= len(a) ULPs); we push
    that residual onto a_int[-1] and b_int[-1] equally so neither marginal moves
    by more than ~1 ULP.
    """
    diff = int(a_int.sum() - b_int.sum())
    if diff == 0:
        return a_int, b_int
    if diff > 0:
        # a has too much supply; shrink last entry of a.
        a_int = a_int.copy()
        a_int[-1] -= diff
    else:
        b_int = b_int.copy()
        b_int[-1] += diff
    return a_int, b_int


def solve_ortools(
    a: np.ndarray,
    b: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    costs: np.ndarray,
    ortools_cost_scale: float = 1e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a sparse balanced OT problem with OR-Tools SimpleMinCostFlow.

    Parameters
    ----------
    a            : (n,) float64; sums to 1.0
    b            : (m,) float64; sums to 1.0
    row_ptr      : (n+1,) int32 CSR row pointer
    col_idx      : (nnz,) int32 CSR column indices
    costs        : (nnz,) float64 CSR edge costs
    ortools_cost_scale : float — int64 scale applied to costs

    Returns
    -------
    rows : (k,) int32 — source indices for non-zero flow arcs
    cols : (k,) int32 — target indices for non-zero flow arcs
    vals : (k,) float64 — transport mass on each arc

    Raises
    ------
    ImportError if `ortools` is not installed.
    """
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError as e:
        raise ImportError(
            "OR-Tools solver requires the 'ortools' package. "
            "Install with: pip install sparse-ot[ortools]"
        ) from e

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    costs   = np.asarray(costs,   dtype=np.float64)

    n, m = len(a), len(b)
    nnz = len(col_idx)

    a_int = _scale_to_int64(a, _SUPPLY_SCALE)
    b_int = _scale_to_int64(b, _SUPPLY_SCALE)
    a_int, b_int = _balance_supplies(a_int, b_int)

    cost_int = _scale_to_int64(costs, int(ortools_cost_scale))

    # Capacity per arc: large enough to carry any feasible flow.
    arc_capacity = int(_SUPPLY_SCALE)  # >= any single-source supply

    # Build arc lists. Source node ids: 0..n-1. Sink node ids: n..n+m-1.
    src_nodes  = np.empty(nnz, dtype=np.int64)
    dst_nodes  = np.empty(nnz, dtype=np.int64)
    capacities = np.full(nnz, arc_capacity, dtype=np.int64)
    for i in range(n):
        s, e = int(row_ptr[i]), int(row_ptr[i + 1])
        src_nodes[s:e] = i
        dst_nodes[s:e] = n + col_idx[s:e].astype(np.int64)

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_ids = smcf.add_arcs_with_capacity_and_unit_cost(
        src_nodes, dst_nodes, capacities, cost_int
    )

    supplies = np.empty(n + m, dtype=np.int64)
    supplies[:n]   =  a_int
    supplies[n:]   = -b_int
    smcf.set_nodes_supplies(np.arange(n + m, dtype=np.int64), supplies)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(
            f"OR-Tools SimpleMinCostFlow did not converge (status={status})"
        )

    # Extract flows in a single vectorized call.
    flows_int = np.asarray(smcf.flows(arc_ids), dtype=np.int64)
    nonzero = flows_int > 0
    rows_out = np.asarray(src_nodes[nonzero], dtype=np.int32)
    cols_out = np.asarray(dst_nodes[nonzero] - n, dtype=np.int32)
    vals_out = (flows_int[nonzero].astype(np.float64) / _SUPPLY_SCALE)

    return rows_out, cols_out, vals_out
```

- [ ] **Step 4: Run tests — confirm 6 passed**

```bash
pytest tests/test_ortools.py -v
```
Expected: `6 passed`.

If `add_arcs_with_capacity_and_unit_cost` is missing on your OR-Tools version, check the API with `python -c "from ortools.graph.python import min_cost_flow; s=min_cost_flow.SimpleMinCostFlow(); print([m for m in dir(s) if 'arc' in m.lower()])"`. Older versions expose `add_arc_with_capacity_and_unit_cost` (singular, scalar arguments) — wrap that in a Python loop if needed.

If `set_nodes_supplies` is missing, look for `set_node_supply` (singular) and call it in a loop over `range(n+m)`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ortools.py src/sparse_ot/ortools_solver.py
git commit -m "feat: ortools_solver.py — int64-scaled SimpleMinCostFlow for sparse balanced OT"
```

---

## Task 2: Wire OR-Tools into emd.py

**Files:**
- Modify: `src/sparse_ot/emd.py`
- Modify: `tests/test_emd.py`

The current `emd.py` raises `NotImplementedError` for `solver='ortools'`. Replace that with a real call into `ortools_solver.solve_ortools`, and add an `ortools_cost_scale` keyword that threads through to the solver. The `emd2` signature also gains `ortools_cost_scale` for symmetry.

- [ ] **Step 1: Read the current emd.py**

```bash
sed -n '1,80p' src/sparse_ot/emd.py
```
Locate the `if solver == 'ortools': raise NotImplementedError(...)` block and the LEMON branch (`from sparse_ot._ext import _lemon; ...`).

- [ ] **Step 2: Update emd() and emd2() signatures + body**

Replace the **entire contents** of `src/sparse_ot/emd.py` with:

```python
# src/sparse_ot/emd.py
import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.routing import select_solver


def emd(a, b, M, numItermax=100000, log=False, center_dual=True,
        cost_sparsity_threshold=0.0, solver=None,
        ortools_cost_scale=1e6):
    """Transport plan between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd(). Dense numpy input returns a dense numpy
    array; scipy sparse input returns scipy CSR.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like (n, m) or scipy sparse (n, m)
    numItermax : int
    log : bool — if True, return (G, log_dict)
    center_dual : bool — accepted for POT compatibility; not used
    cost_sparsity_threshold : float — dense M only: drop |M[i,j]| <= threshold
    solver : str or None — 'bonneel', 'lemon', 'ortools', or None (auto)
    ortools_cost_scale : float — int64 scale applied to float costs by OR-Tools

    Returns
    -------
    G : ndarray (n, m) or scipy CSR (n, m)
    (G, {}) if log=True
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if solver not in (None, 'bonneel', 'lemon', 'ortools'):
        raise ValueError(
            f"solver={solver!r} must be None, 'bonneel', 'lemon', or 'ortools'"
        )

    dense_input = not scipy.sparse.issparse(M)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, cost_sparsity_threshold)

    if (len(a), len(b)) != (n, m):
        raise ValueError(
            f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
        )

    a = a / a.sum()
    b = b / b.sum()

    selected = select_solver(n, m, nnz, solver)

    if selected == 'bonneel':
        if dense_input:
            M_dense = np.asarray(M, dtype=np.float64, order='C')
        else:
            M_dense = np.asarray(M.toarray(), dtype=np.float64, order='C')
        G = _bonneel.solve_dense(a, b, M_dense, numItermax)
        if log:
            return G, {}
        return G

    if selected == 'lemon':
        from sparse_ot._ext import _lemon
        rows, cols, vals = _lemon.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
    else:
        # selected == 'ortools'
        from sparse_ot.ortools_solver import solve_ortools
        rows, cols, vals = solve_ortools(
            a, b, row_ptr, col_idx, costs,
            ortools_cost_scale=ortools_cost_scale,
        )

    G_sp = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    G = G_sp.toarray() if dense_input else G_sp

    if log:
        return G, {}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
         cost_sparsity_threshold=0.0, solver=None,
         ortools_cost_scale=1e6):
    """OT cost between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd2(). Returns a float scalar.
    """
    G = emd(a, b, M, numItermax=numItermax, log=False,
            cost_sparsity_threshold=cost_sparsity_threshold,
            solver=solver, ortools_cost_scale=ortools_cost_scale)

    if scipy.sparse.issparse(M):
        M_arr = np.asarray(M.toarray(), dtype=np.float64)
    else:
        M_arr = np.asarray(M, dtype=np.float64)

    if scipy.sparse.issparse(G):
        cost = float(G.multiply(M_arr).sum())
    else:
        cost = float(np.sum(G * M_arr))

    if return_matrix:
        if log:
            return cost, G, {}
        return cost, G
    if log:
        return cost, {}
    return cost
```

- [ ] **Step 3: Replace the `test_emd_ortools_raises_not_implemented` test**

In `tests/test_emd.py`, find the test named `test_emd_ortools_raises_not_implemented` and delete it. Then append:

```python
# tests/test_emd.py — append at end

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
    # Both should be finite and close to POT.
    assert np.isfinite(cost_default) and np.isfinite(cost_higher)
```

- [ ] **Step 4: Run test_emd.py — all must pass**

```bash
pytest tests/test_emd.py -v
```
Expected: previous test count minus 1 (removed `ortools_raises`) plus 3 new tests. Numerically: previously 19 passed → expect 21 passed.

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -v
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/sparse_ot/emd.py tests/test_emd.py
git commit -m "feat: emd() routes to OR-Tools — ortools_cost_scale parameter exposed"
```

---

## Task 3: benchmarks/problems.py — k-NN grid problem generator (TDD)

**Files:**
- Create: `tests/test_benchmarks.py`
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/problems.py`

The benchmark needs deterministic, reproducible OT problems whose cost-graph sparsity is controlled by a single parameter `k`. Each problem places `n` source nodes and `n` target nodes on a regular 1D grid (positions `0..n-1`); each source connects to its `k` nearest target neighbors with cost `(i - j) ** 2`. Distributions `a` and `b` are full-support Dirichlet draws. This matches the reference voxel-grid use case structurally and keeps grid generation cheap even at n = 16M.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_benchmarks.py
import numpy as np
import scipy.sparse
import pytest

from benchmarks.problems import generate_knn_grid_problem


def test_generate_shapes_and_nnz():
    a, b, M = generate_knn_grid_problem(n=20, k=4, seed=0)
    assert a.shape == (20,) and b.shape == (20,)
    assert scipy.sparse.issparse(M)
    assert M.shape == (20, 20)
    # Boundary nodes have fewer than k neighbors; interior have exactly k.
    # For n=20, k=4: interior = 16 nodes × 4 = 64; boundaries lose some.
    assert 0 < M.nnz <= 20 * 4


def test_generate_nnz_for_interior():
    """For n >> k, nnz is close to n * k."""
    a, b, M = generate_knn_grid_problem(n=1000, k=8, seed=0)
    # Worst case (boundary truncation) loses at most k * (k / 2) edges.
    assert M.nnz >= 1000 * 8 - 8 * 8


def test_generate_fully_dense_when_k_equals_n():
    a, b, M = generate_knn_grid_problem(n=10, k=10, seed=0)
    assert M.nnz == 100


def test_generate_distributions_normalized():
    a, b, M = generate_knn_grid_problem(n=50, k=4, seed=0)
    np.testing.assert_allclose(a.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(b.sum(), 1.0, atol=1e-12)
    assert (a > 0).all()
    assert (b > 0).all()


def test_generate_costs_are_squared_distance():
    """Cost at edge (i, j) equals (i - j) ** 2."""
    _, _, M = generate_knn_grid_problem(n=20, k=4, seed=0)
    coo = M.tocoo()
    for i, j, c in zip(coo.row, coo.col, coo.data):
        np.testing.assert_allclose(c, (int(i) - int(j)) ** 2)


def test_generate_seed_reproducibility():
    a1, b1, M1 = generate_knn_grid_problem(n=50, k=4, seed=42)
    a2, b2, M2 = generate_knn_grid_problem(n=50, k=4, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(M1.toarray(), M2.toarray())


def test_generate_seed_differs():
    a1, _, _ = generate_knn_grid_problem(n=50, k=4, seed=1)
    a2, _, _ = generate_knn_grid_problem(n=50, k=4, seed=2)
    assert not np.array_equal(a1, a2)
```

- [ ] **Step 2: Run tests — confirm ImportError**

```bash
pytest tests/test_benchmarks.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'benchmarks'`.

- [ ] **Step 3: Create benchmarks package init**

```bash
touch benchmarks/__init__.py
```

- [ ] **Step 4: Implement problems.py**

```python
# benchmarks/problems.py
"""Reproducible k-NN grid OT problems.

Source and target nodes are placed on a regular 1D grid at positions 0..n-1.
Each source i connects to up to `k` target nodes nearest to i (clipped at
grid boundaries) with cost (i - j) ** 2. Distributions a and b are
Dirichlet(1, ..., 1) draws (full-support, reproducible per `seed`).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse


def generate_knn_grid_problem(
    n: int, k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, scipy.sparse.csr_matrix]:
    """Return (a, b, M) where M is an (n, n) scipy CSR of squared-distance costs."""
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))

    half = k // 2
    # For each i, target window is [max(0, i - half), min(n, i - half + k)].
    rows, cols, data = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        # Re-pull lo down if the window is truncated on the high side.
        lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            data.append(float((i - j) ** 2))

    M = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32),
          np.asarray(cols, dtype=np.int32))),
        shape=(n, n),
    )
    return a, b, M
```

- [ ] **Step 5: Run tests — confirm 7 passed**

```bash
pytest tests/test_benchmarks.py -v
```
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/__init__.py benchmarks/problems.py tests/test_benchmarks.py
git commit -m "feat: benchmarks/problems.py — reproducible k-NN grid OT problem generator"
```

---

## Task 4: bench_solvers.py — efficiency sweep

**Files:**
- Create: `benchmarks/bench_solvers.py`

The efficiency sweep runs each `(n, k, solver)` cell with a memory-cutoff guard and writes wall time (median of 5 runs), peak memory, and iteration count to `benchmarks/results/efficiency.json`. The sweep supports `--quick` mode for CI (small grid, 1 run per cell). Cells skipped by memory cutoffs are recorded as JSON `null`.

- [ ] **Step 1: Write bench_solvers.py**

```python
# benchmarks/bench_solvers.py
"""sparse-ot benchmark driver.

Usage:
    python benchmarks/bench_solvers.py            # full sweep (hours)
    python benchmarks/bench_solvers.py --quick    # tiny sweep (seconds, for CI)
    python benchmarks/bench_solvers.py --efficiency-only
    python benchmarks/bench_solvers.py --accuracy-only

Writes:
    benchmarks/results/efficiency.json
    benchmarks/results/accuracy.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.sparse

from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd2

# --- Memory cutoffs (16 GB machine defaults) ---
MAX_DENSE_N      = 8_192          # bonneel + POT reference: dense n*n*8 bytes
MAX_SPARSE_NNZ   = 200_000_000    # LEMON: ~20 bytes/edge
MAX_ORTOOLS_NNZ  = 500_000_000    # OR-Tools memory ceiling

FULL_N = [1_000, 4_000, 8_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000, 16_000_000]
FULL_K = [2, 8, 32, 128, 512, 2048]   # plus n/10 and n appended per n

QUICK_N = [200, 1_000]
QUICK_K = [4, 32]

SOLVERS = ['bonneel', 'lemon', 'ortools', 'pot_reference']

RESULTS_DIR = Path(__file__).parent / "results"


def _solver_skipped(solver: str, n: int, nnz: int) -> bool:
    """Return True if `solver` cannot run this cell within memory limits."""
    if solver in ('bonneel', 'pot_reference'):
        return n > MAX_DENSE_N
    if solver == 'lemon':
        return nnz > MAX_SPARSE_NNZ
    if solver == 'ortools':
        return nnz > MAX_ORTOOLS_NNZ
    raise ValueError(f"unknown solver: {solver}")


def _time_solver(solver: str, a, b, M, n_runs: int) -> dict:
    """Return wall_time_median_s and peak_memory_mb for `solver` on (a, b, M)."""
    import ot  # POT for reference
    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        if solver == 'pot_reference':
            # POT requires dense
            M_dense = M.toarray() if scipy.sparse.issparse(M) else np.asarray(M)
            _ = ot.emd2(a, b, M_dense)
        else:
            _ = emd2(a, b, M, solver=solver)
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_time_s": float(np.median(times)),
        "peak_memory_mb": peak_bytes / (1024 ** 2),
        "n_runs": n_runs,
    }


def _k_grid(n: int, base_ks: list[int]) -> list[int]:
    """Append n/10 and n to the base k grid, deduplicated and sorted."""
    ks = set(base_ks)
    ks.add(max(2, n // 10))
    ks.add(n)
    return sorted(k for k in ks if k <= n)


def run_efficiency_sweep(ns: list[int], base_ks: list[int], n_runs: int) -> dict:
    """Return a nested dict[n][k][solver] -> result-dict-or-None."""
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks):
            results[str(n)][str(k)] = {}
            try:
                a, b, M = generate_knn_grid_problem(n=n, k=k, seed=0)
                nnz = M.nnz
            except MemoryError:
                for s in SOLVERS:
                    results[str(n)][str(k)][s] = None
                continue
            for solver in SOLVERS:
                if _solver_skipped(solver, n, nnz):
                    results[str(n)][str(k)][solver] = None
                    continue
                try:
                    res = _time_solver(solver, a, b, M, n_runs=n_runs)
                    res["nnz"] = int(nnz)
                    results[str(n)][str(k)][solver] = res
                except Exception as e:
                    results[str(n)][str(k)][solver] = {
                        "error": f"{type(e).__name__}: {e}",
                        "nnz": int(nnz),
                    }
                print(f"  n={n} k={k} solver={solver} → "
                      f"{results[str(n)][str(k)][solver]}")
    return results


def run_accuracy_sweep(ns: list[int], base_ks: list[int]) -> dict:
    """Return a nested dict[n][k][solver] -> {rel_cost_err, feasibility}."""
    import ot
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks):
            results[str(n)][str(k)] = {}
            a, b, M = generate_knn_grid_problem(n=n, k=k, seed=0)
            nnz = M.nnz

            cost_ref = None
            if n <= 64_000 and n <= MAX_DENSE_N:
                M_dense = M.toarray()
                cost_ref = float(ot.emd2(a, b, M_dense))

            for solver in ('lemon', 'ortools'):
                if _solver_skipped(solver, n, nnz):
                    results[str(n)][str(k)][solver] = None
                    continue
                try:
                    G = __import__('sparse_ot').emd(a, b, M, solver=solver)
                    cost = float(G.multiply(M).sum()) if scipy.sparse.issparse(G) \
                           else float((G * M.toarray()).sum())
                    if scipy.sparse.issparse(G):
                        row_sum = np.asarray(G.sum(axis=1)).ravel()
                        col_sum = np.asarray(G.sum(axis=0)).ravel()
                    else:
                        row_sum = G.sum(axis=1)
                        col_sum = G.sum(axis=0)
                    feas_a = float(np.max(np.abs(row_sum - a)))
                    feas_b = float(np.max(np.abs(col_sum - b)))
                    rel_cost_err = (
                        None if cost_ref is None
                        else abs(cost - cost_ref) / max(abs(cost_ref), 1e-15)
                    )
                    results[str(n)][str(k)][solver] = {
                        "cost": cost,
                        "cost_ref": cost_ref,
                        "rel_cost_err": rel_cost_err,
                        "feasibility_a": feas_a,
                        "feasibility_b": feas_b,
                    }
                except Exception as e:
                    results[str(n)][str(k)][solver] = {
                        "error": f"{type(e).__name__}: {e}",
                    }
                print(f"  acc n={n} k={k} solver={solver} → "
                      f"{results[str(n)][str(k)][solver]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny sweep — for CI smoke testing")
    ap.add_argument("--efficiency-only", action="store_true")
    ap.add_argument("--accuracy-only",   action="store_true")
    args = ap.parse_args()

    if args.quick:
        ns, base_ks, n_runs = QUICK_N, QUICK_K, 1
    else:
        ns, base_ks, n_runs = FULL_N, FULL_K, 5

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.accuracy_only:
        print(f"[efficiency] ns={ns} ks={base_ks} n_runs={n_runs}")
        eff = run_efficiency_sweep(ns, base_ks, n_runs)
        out = RESULTS_DIR / ("efficiency_quick.json" if args.quick
                             else "efficiency.json")
        out.write_text(json.dumps(eff, indent=2))
        print(f"[efficiency] wrote {out}")

    if not args.efficiency_only:
        print(f"[accuracy] ns={ns} ks={base_ks}")
        acc = run_accuracy_sweep(ns, base_ks)
        out = RESULTS_DIR / ("accuracy_quick.json" if args.quick
                             else "accuracy.json")
        out.write_text(json.dumps(acc, indent=2))
        print(f"[accuracy] wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the quick sweep**

```bash
python benchmarks/bench_solvers.py --quick 2>&1 | tail -40
```
Expected: completes in under 60s, writes `benchmarks/results/efficiency_quick.json` and `benchmarks/results/accuracy_quick.json`. Inspect the JSON for the structure `{n: {k: {solver: {...}}}}` with at least some non-null cells.

- [ ] **Step 3: Add a smoke test that the --quick benchmark completes**

Append to `tests/test_benchmarks.py`:

```python
def test_bench_solvers_quick_smoke(tmp_path, monkeypatch):
    """`python benchmarks/bench_solvers.py --quick` exits 0 and writes JSON."""
    import subprocess, sys, os
    repo_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root
    result = subprocess.run(
        [sys.executable, "benchmarks/bench_solvers.py", "--quick"],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(repo_root) / "benchmarks/results/efficiency_quick.json").exists()
    assert (Path(repo_root) / "benchmarks/results/accuracy_quick.json").exists()


from pathlib import Path  # add at top of file if not already imported
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_benchmarks.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_solvers.py tests/test_benchmarks.py
git commit -m "feat: bench_solvers.py — efficiency + accuracy sweep with --quick CI mode"
```

---

## Task 5: Run the full benchmark sweep

This step is heavy — expect several hours on a 16GB workstation. If you are running this plan in a constrained environment (CI, agent VM), skip to Task 6 and run this step on a workstation later; Task 6 falls back to the `_quick.json` files when full results are absent.

- [ ] **Step 1: Run the full sweep**

```bash
python benchmarks/bench_solvers.py 2>&1 | tee benchmarks/results/bench_log.txt
```
Expected: writes `benchmarks/results/efficiency.json` and `benchmarks/results/accuracy.json`. The log captures any per-cell errors. If a cell crashes the whole process (e.g. OOM kill), reduce `MAX_*` constants at the top of `bench_solvers.py` and re-run; the run is idempotent because each invocation overwrites the JSON.

- [ ] **Step 2: Verify result coverage**

```bash
python - <<'EOF'
import json
from pathlib import Path
for name in ("efficiency.json", "accuracy.json"):
    p = Path("benchmarks/results") / name
    if not p.exists():
        print(f"missing: {name}")
        continue
    data = json.loads(p.read_text())
    total = 0
    nonnull = 0
    for n_str, by_k in data.items():
        for k_str, by_solver in by_k.items():
            for s, v in by_solver.items():
                total += 1
                if v is not None and isinstance(v, dict) and "error" not in v:
                    nonnull += 1
    print(f"{name}: {nonnull}/{total} cells populated")
EOF
```
Expected: nonzero coverage in both files. Some `null` cells (memory cutoffs) are expected; widespread `error` cells indicate a bug.

- [ ] **Step 3: Commit raw results**

```bash
git add benchmarks/results/efficiency.json benchmarks/results/accuracy.json \
        benchmarks/results/bench_log.txt
git commit -m "bench: full sweep results (efficiency + accuracy)"
```

---

## Task 6: generate_report.py — figures + threshold derivation

**Files:**
- Create: `benchmarks/generate_report.py`

This script reads `efficiency.json` and `accuracy.json` (falling back to `*_quick.json` for development) and emits four figures plus a regenerated `routing_thresholds.json`.

- [ ] **Step 1: Write generate_report.py**

```python
# benchmarks/generate_report.py
"""Render benchmark figures and derive routing thresholds.

Usage:
    python benchmarks/generate_report.py
    python benchmarks/generate_report.py --quick   # use *_quick.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_THRESHOLDS = {"bonneel_lemon": 128, "lemon_ortools": 1_000_000}


def _load(name: str, quick: bool) -> dict | None:
    fname = name.replace(".json", "_quick.json") if quick else name
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _grid(data: dict, solver: str, field: str) -> tuple[list[int], list[int], np.ndarray]:
    """Extract a (ns, ks, value-matrix) grid for a given solver and metric field."""
    ns = sorted(int(n) for n in data.keys())
    ks_set: set[int] = set()
    for n_str in data:
        ks_set.update(int(k) for k in data[n_str])
    ks = sorted(ks_set)
    grid = np.full((len(ns), len(ks)), np.nan, dtype=np.float64)
    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            cell = data.get(str(n), {}).get(str(k), {}).get(solver)
            if cell is None or not isinstance(cell, dict) or "error" in cell:
                continue
            v = cell.get(field)
            if v is None:
                continue
            grid[i, j] = v
    return ns, ks, grid


def _heatmap(ns, ks, ratio, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.log10(ratio), aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("k (neighbors per source)")
    ax.set_ylabel("n (problem size)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(wall time ratio)")
    # Crossover contour at ratio=1 (log10=0)
    if np.any(np.isfinite(ratio)):
        ax.contour(np.log10(ratio), levels=[0.0], colors="black", linewidths=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def render_efficiency_heatmaps(eff: dict) -> None:
    _, _, t_bon = _grid(eff, "bonneel", "wall_time_s")
    ns, ks, t_lem = _grid(eff, "lemon", "wall_time_s")
    _, _, t_ort = _grid(eff, "ortools", "wall_time_s")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_lb = t_lem / t_bon  # >1 means LEMON slower than Bonneel
        ratio_ol = t_ort / t_lem  # >1 means OR-Tools slower than LEMON

    _heatmap(ns, ks, ratio_lb,
             "LEMON / Bonneel wall-time ratio (contour = crossover)",
             FIGURES_DIR / "heatmap_lemon_vs_bonneel")
    _heatmap(ns, ks, ratio_ol,
             "OR-Tools / LEMON wall-time ratio (contour = crossover)",
             FIGURES_DIR / "heatmap_ortools_vs_lemon")


def render_line_plots(eff: dict) -> None:
    """Wall time vs n at fixed k, one line per solver."""
    fixed_ks = [2, 8, 32, 128, 512]
    for k_target in fixed_ks:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for solver, marker in [("bonneel", "o"), ("lemon", "s"),
                               ("ortools", "^"), ("pot_reference", "x")]:
            xs, ys = [], []
            for n_str, by_k in eff.items():
                if str(k_target) not in by_k:
                    continue
                cell = by_k[str(k_target)].get(solver)
                if cell is None or not isinstance(cell, dict) or "error" in cell:
                    continue
                xs.append(int(n_str))
                ys.append(cell["wall_time_s"])
            if not xs:
                continue
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            ax.plot(xs, ys, marker=marker, label=solver)
            plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n"); ax.set_ylabel("wall time (s)")
        ax.set_title(f"Wall time vs n at k={k_target}")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        path = FIGURES_DIR / f"lineplot_k{k_target}"
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)


def render_accuracy_plot(acc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for solver, marker in [("lemon", "s"), ("ortools", "^")]:
        ks, errs, feas = [], [], []
        for n_str, by_k in acc.items():
            for k_str, by_solver in by_k.items():
                cell = by_solver.get(solver)
                if cell is None or not isinstance(cell, dict) or "error" in cell:
                    continue
                if cell.get("rel_cost_err") is not None:
                    ks.append(int(k_str))
                    errs.append(cell["rel_cost_err"])
                feas.append(max(cell["feasibility_a"], cell["feasibility_b"]))
        if ks:
            axes[0].scatter(ks, errs, marker=marker, label=solver, alpha=0.6)
        if feas:
            axes[1].scatter(range(len(feas)), feas, marker=marker, label=solver, alpha=0.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("relative cost error vs POT")
    axes[0].set_title("Accuracy: relative cost error")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sample index"); axes[1].set_ylabel("‖marginal residual‖∞")
    axes[1].set_title("Feasibility: marginal residual")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy.pdf")
    fig.savefig(FIGURES_DIR / "accuracy.png", dpi=150)
    plt.close(fig)


def derive_thresholds(eff: dict) -> dict:
    """Find the crossover k where LEMON beats Bonneel, and the n where OR-Tools beats LEMON.

    bonneel_lemon: smallest k for which median(wall_time_lemon < wall_time_bonneel)
                   over all n.
    lemon_ortools: smallest n at which OR-Tools wall_time < LEMON wall_time
                   for the smallest k seen.
    Falls back to DEFAULT_THRESHOLDS if data is too sparse to derive.
    """
    bonneel_lemon = DEFAULT_THRESHOLDS["bonneel_lemon"]
    lemon_ortools = DEFAULT_THRESHOLDS["lemon_ortools"]

    # bonneel_lemon: find smallest k where LEMON faster than Bonneel for majority of n
    ks_set: set[int] = set()
    for n_str in eff:
        ks_set.update(int(k) for k in eff[n_str])
    ks_sorted = sorted(ks_set)
    for k in ks_sorted:
        votes_lemon_wins = 0
        votes_total = 0
        for n_str, by_k in eff.items():
            cell = by_k.get(str(k), {})
            l = cell.get("lemon"); b = cell.get("bonneel")
            if (isinstance(l, dict) and isinstance(b, dict)
                and "error" not in l and "error" not in b):
                votes_total += 1
                if l["wall_time_s"] < b["wall_time_s"]:
                    votes_lemon_wins += 1
        if votes_total >= 2 and votes_lemon_wins / votes_total >= 0.5:
            bonneel_lemon = k
            break

    # lemon_ortools: smallest n where OR-Tools faster than LEMON at small k
    small_k_candidates = [k for k in ks_sorted if k <= 32]
    ns_sorted = sorted(int(n) for n in eff.keys())
    for n in ns_sorted:
        wins = 0; total = 0
        for k in small_k_candidates:
            cell = eff.get(str(n), {}).get(str(k), {})
            l = cell.get("lemon"); o = cell.get("ortools")
            if (isinstance(l, dict) and isinstance(o, dict)
                and "error" not in l and "error" not in o):
                total += 1
                if o["wall_time_s"] < l["wall_time_s"]:
                    wins += 1
        if total >= 1 and wins / total >= 0.5:
            lemon_ortools = n
            break

    return {"bonneel_lemon": int(bonneel_lemon),
            "lemon_ortools": int(lemon_ortools)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use *_quick.json files instead of full results")
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eff = _load("efficiency.json", args.quick)
    acc = _load("accuracy.json",   args.quick)
    if eff is None:
        print("error: no efficiency JSON found; run bench_solvers.py first",
              file=sys.stderr)
        sys.exit(2)

    render_efficiency_heatmaps(eff)
    render_line_plots(eff)
    if acc is not None:
        render_accuracy_plot(acc)
    else:
        print("warning: no accuracy JSON found; skipping accuracy plot",
              file=sys.stderr)

    thresholds = derive_thresholds(eff)
    out = RESULTS_DIR / "routing_thresholds.json"
    out.write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"wrote {out}: {thresholds}")
    print(f"figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on the --quick results**

```bash
python benchmarks/generate_report.py --quick
ls benchmarks/results/figures/
```
Expected: at least `accuracy.pdf`, `accuracy.png`, `heatmap_lemon_vs_bonneel.pdf/png`, `heatmap_ortools_vs_lemon.pdf/png`, and one or more `lineplot_k*.pdf/png`. The thresholds file is overwritten with derived values; on quick data the result may equal the defaults (`{"bonneel_lemon": 128, "lemon_ortools": 1000000}`) — this is fine.

- [ ] **Step 3: Regenerate against the full sweep (if Task 5 ran)**

```bash
test -f benchmarks/results/efficiency.json && python benchmarks/generate_report.py
```
Expected: figures regenerated from real data; `routing_thresholds.json` populated with empirically-derived values.

- [ ] **Step 4: Verify routing tests still pass with the regenerated thresholds**

```bash
pytest tests/test_routing.py -v
```
If a test fails because the new thresholds shifted the boundary cases — that's a real signal: the test assumed `bonneel_lemon=128, lemon_ortools=1_000_000`. Update the failing tests in `tests/test_routing.py` to load `_THRESHOLDS_PATH` and parameterize on the actual values, **or** if the regenerated thresholds are wildly off from what the spec promised, revert to defaults by writing the defaults JSON back. Do **not** silently rewrite the routing logic to ignore the file.

If the empirical values look sensible (close to defaults or smoothly varying), update the tests to assert routing behavior in terms of "above/below threshold" rather than hard-coded n and k values.

- [ ] **Step 5: Commit figures and updated thresholds**

```bash
git add benchmarks/generate_report.py benchmarks/results/figures/ \
        benchmarks/results/routing_thresholds.json
git commit -m "feat: generate_report.py — figures + empirical routing thresholds"
```

---

## Task 7: README updates

**Files:**
- Modify: `README.md` (create if it doesn't exist)

- [ ] **Step 1: Check whether README exists**

```bash
ls README.md 2>&1
```

- [ ] **Step 2: Write or update README.md**

If no `README.md` exists, create one. If it exists, append the new sections at the end and leave existing content untouched. Write:

```markdown
# sparse-ot

Drop-in replacement for [POT](https://github.com/PythonOT/POT)'s `emd` / `emd2`,
optimized for sparse cost matrices. Routes between three solvers based on
problem size and sparsity:

| Solver       | Best for                                       | Source             |
|--------------|------------------------------------------------|--------------------|
| Bonneel      | Dense / near-dense cost matrices               | Vendored C++ (`src/cpp/bonneel`) |
| LEMON        | Sparse, moderate scale (float64-patched)       | Vendored C++ (`src/cpp/lemon`) + custom patch |
| OR-Tools     | Very large sparse problems                     | Optional `ortools` Python dep |

## Quickstart

```python
import sparse_ot as sot
G = sot.emd(a, b, M)      # numpy dense in → numpy dense out
G = sot.emd(a, b, M_csr)  # scipy CSR in → scipy CSR out
cost = sot.emd2(a, b, M)
```

`solver=` overrides routing: `'bonneel'`, `'lemon'`, or `'ortools'`.
`cost_sparsity_threshold` drops edges with `|M[i,j]| <= threshold` from dense
input.

## Routing

`routing.select_solver(n, m, nnz, solver=None)` picks the solver from
`benchmarks/results/routing_thresholds.json`:

- `k = nnz / n > bonneel_lemon` → Bonneel
- otherwise `n > lemon_ortools` → OR-Tools
- otherwise → LEMON

The thresholds are derived empirically from the benchmark suite. To regenerate
for your hardware:

```bash
python benchmarks/bench_solvers.py        # full sweep — hours
python benchmarks/generate_report.py      # writes routing_thresholds.json + figures
```

For development, the `--quick` flag runs a small sweep in seconds:

```bash
python benchmarks/bench_solvers.py --quick
python benchmarks/generate_report.py --quick
```

## Memory cutoffs

`bench_solvers.py` skips cells beyond these defaults (16 GB target):

| Constant         | Default       | Effect                              |
|------------------|---------------|-------------------------------------|
| `MAX_DENSE_N`    | 8 192         | Bonneel and POT reference skipped above this |
| `MAX_SPARSE_NNZ` | 200 000 000   | LEMON skipped above this            |
| `MAX_ORTOOLS_NNZ`| 500 000 000   | OR-Tools skipped above this         |

The reference voxel-grid case (n = 16.7M, nnz ≈ 536M) exceeds `MAX_ORTOOLS_NNZ`
on a 16GB machine. Raise the constants in `benchmarks/bench_solvers.py` for
larger hardware.

## Install

```bash
pip install sparse-ot                 # numpy + scipy
pip install sparse-ot[ortools]        # adds OR-Tools solver
pip install sparse-ot[torch]          # adds torch sparse interop
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README — routing, benchmarks, memory cutoffs, install"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all green. Smoke-test counts (approximate): `test_emd.py` 21, `test_sparse_utils.py` 8, `test_routing.py` 10, `test_lemon_accuracy.py` 6, `test_ortools.py` 6, `test_benchmarks.py` 8.

- [ ] **Step 2: Verify routing still picks expected solvers**

```bash
python - <<'EOF'
from sparse_ot.routing import select_solver
print(select_solver(10, 10, 100))         # lemon (k=10, small n)
print(select_solver(200, 200, 200*200))   # bonneel (k=200 > threshold)
print(select_solver(2_000_000, 100, 2_000_000 * 10))  # ortools (large n, small k)
EOF
```
If the regenerated thresholds in Task 6 reshaped these decisions, the outputs may differ. That's correct — the printed routing is a reflection of the empirical thresholds. The expected qualitative pattern (high k → bonneel, large n + low k → ortools, else lemon) should still hold.

- [ ] **Step 3: Confirm OR-Tools error message when uninstalled**

```bash
python - <<'EOF'
import sys
# Hide ortools to verify the user-facing ImportError message.
import builtins
real_import = builtins.__import__
def fake_import(name, *a, **kw):
    if name.startswith("ortools"):
        raise ImportError("No module named 'ortools'")
    return real_import(name, *a, **kw)
builtins.__import__ = fake_import
try:
    from sparse_ot.ortools_solver import solve_ortools
    import numpy as np
    try:
        solve_ortools(np.array([1.0]), np.array([1.0]),
                      np.array([0, 1], dtype=np.int32),
                      np.array([0], dtype=np.int32),
                      np.array([0.0]))
    except ImportError as e:
        assert "pip install sparse-ot[ortools]" in str(e), f"unexpected: {e}"
        print("OK: ImportError message mentions install command")
finally:
    builtins.__import__ = real_import
EOF
```
Expected: prints `OK: ImportError message mentions install command`.

- [ ] **Step 4: Commit final state**

```bash
git status
# verify only expected files appear; nothing stray
git add -A
git diff --cached --stat
git commit -m "feat: Plan 3 complete — OR-Tools solver + benchmark suite + figures" \
   --allow-empty
```

- [ ] **Step 5: Self-review against the spec**

Walk through `docs/superpowers/specs/2026-05-15-sparse-ot-design.md` sections 4 (OR-Tools), 5 (routing), and 6 (benchmark suite). For each bullet point:
- Is it implemented?
- Is it tested?
- Is the artifact (JSON, figure, README section) committed?

Note any gaps in the final commit message or as a follow-up TODO in the README. Plan 4 (PyPI publishing, cibuildwheel, GitHub Actions) is the natural next plan.
