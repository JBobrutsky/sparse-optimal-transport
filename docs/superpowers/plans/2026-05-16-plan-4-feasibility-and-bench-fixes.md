# Feasibility Check + Benchmark Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sparse path of `sparse_ot` correct end-to-end: add the input-feasibility contract from spec §2/§3, fix the benchmark generator to produce feasible-by-construction instances, replace the broken POT-as-reference accuracy harness with min-cost-across-solvers (spec §6), and re-derive routing thresholds from a clean sweep.

**Architecture:** A new `sparse_ot/feasibility.py` module implements `InfeasibleProblemError` and an O(nnz·α) union-find feasibility check, called from `emd.py` before solver dispatch on sparse-input paths. The benchmark generator is hardened to assert the identity backbone and float-exact marginal balance. The accuracy harness in `bench_solvers.py` is rewritten to compute `cost_ref` per instance as the min over solvers that ran successfully and were primal-feasible. Once correctness is restored, `generate_report.py` derives routing thresholds from the empirical crossover and writes `routing_thresholds.json`.

**Tech Stack:** Python 3.10+, numpy, scipy.sparse, pybind11 C++ extensions (no changes), pytest.

---

## File Structure

- **Create:** `src/sparse_ot/feasibility.py` — `InfeasibleProblemError`, `check_feasibility(a, b, row_ptr, col_idx)`
- **Create:** `tests/test_feasibility_check.py` — exception contract, diagnostic attributes, skip paths
- **Modify:** `src/sparse_ot/__init__.py` — export `InfeasibleProblemError`
- **Modify:** `src/sparse_ot/emd.py:44-54` — call feasibility check between `to_csr` and `select_solver` on sparse path
- **Modify:** `benchmarks/problems.py` — assert self-edge `(i, i)` present per row; normalize `a`, `b` so `a.sum() == b.sum()` exactly in float64
- **Modify:** `benchmarks/bench_solvers.py` — accuracy harness uses `cost_ref = min(cost_solver for feasible solvers)`, drops POT-as-reference at sparse `k`
- **Modify:** `benchmarks/generate_report.py` — write `routing_thresholds.json` from empirical crossover; update accuracy plot legend
- **Touch (investigation only):** `src/sparse_ot/_ext` (LEMON wrapper), `src/sparse_ot/ortools_solver.py` — root-cause why both returned `INFEASIBLE` on band graphs that *are* feasible

---

## Task 1: Reproduce and root-cause the INFEASIBLE results

**Files:**
- Inspect: `src/sparse_ot/ortools_solver.py`, `src/cpp/lemon_solver.cpp`
- Test: `tests/test_feasibility_root_cause.py` (temporary; delete in Task 7)

- [ ] **Step 1: Write a failing reproducer test**

```python
# tests/test_feasibility_root_cause.py
import numpy as np
import pytest
from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd

@pytest.mark.parametrize("n,k", [(200, 4), (200, 20), (1000, 4), (1000, 32)])
def test_band_graph_is_feasible(n, k):
    a, b, M = generate_knn_grid_problem(n, k, seed=0)
    # band k-NN on the same 1D grid is connected → feasible
    G = emd(a, b, M, solver='lemon')
    assert G.sum() == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run it; record the actual failure mode**

```bash
pytest tests/test_feasibility_root_cause.py -v 2>&1 | tee /tmp/reproducer.log
```

Expected: matches the `efficiency_quick.json` errors — `LEMON CostScaling did not reach OPTIMAL (status=INFEASIBLE)` and/or the analogous OR-Tools error. If the reproducer *passes*, the bug is in the benchmark harness (Task 6 territory), not the solvers — re-scope this task accordingly.

- [ ] **Step 3: Inspect float-exact marginal balance in the generator**

```python
# scratch
import numpy as np
from benchmarks.problems import generate_knn_grid_problem
a, b, _ = generate_knn_grid_problem(200, 4, seed=0)
print(f"sum(a)={a.sum():.20e}  sum(b)={b.sum():.20e}  diff={a.sum()-b.sum():.20e}")
print(f"a.sum() == 1.0 ? {a.sum() == 1.0}")
```

Expected: small but non-zero `a.sum() - b.sum()` (Dirichlet draws normalize to ≈1.0, not exactly 1.0). Confirms or rules out the float-exact balance hypothesis. Note the observed magnitude.

- [ ] **Step 4: Inspect what the wrapper passes to LEMON**

Read `src/cpp/lemon_solver.cpp` and `src/sparse_ot/ortools_solver.py`. Look for: (i) supply/demand integer scaling, (ii) whether `sum(supply) == sum(-demand)` is enforced before solve, (iii) whether the cost-scaling tolerance from the float64 patch is being read correctly. Document findings in the commit message of Step 7.

- [ ] **Step 5: Form a single specific hypothesis**

Write the hypothesis as a one-line comment at the top of `tests/test_feasibility_root_cause.py`. Examples (only one will be correct):
```python
# HYPOTHESIS: sum(a) != sum(b) in float64 → LEMON rejects as infeasible
# HYPOTHESIS: cost-scaling tolerance never read; epsilon < 1 still applied
# HYPOTHESIS: ortools int64 scaling rounds total supply != total demand
```

- [ ] **Step 6: Verify the hypothesis with a minimal fix in a scratch branch**

Apply the minimal change that the hypothesis predicts will fix it (e.g., renormalize `b` so `b.sum() == a.sum()` exactly; or fix the epsilon condition). Re-run the reproducer.

Expected: test from Step 1 passes.

- [ ] **Step 7: Commit findings (not the fix yet)**

```bash
git add tests/test_feasibility_root_cause.py
git commit -m "test: reproducer for sparse-solver INFEASIBLE on feasible band graphs

Documents root cause: <one-line summary from Step 5>.
Fix lands in Task 7."
```

---

## Task 2: Define `InfeasibleProblemError`

**Files:**
- Create: `src/sparse_ot/feasibility.py`
- Test: `tests/test_feasibility_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feasibility_check.py
import pytest
from sparse_ot.feasibility import InfeasibleProblemError

def test_exception_attributes():
    err = InfeasibleProblemError(
        imbalance=0.25,
        component_sources=[0, 1, 2],
        component_targets=[5, 6],
    )
    assert isinstance(err, ValueError)
    assert err.imbalance == 0.25
    assert err.component_sources == [0, 1, 2]
    assert err.component_targets == [5, 6]
    assert "0.25" in str(err)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_feasibility_check.py::test_exception_attributes -v
```

Expected: `ModuleNotFoundError: No module named 'sparse_ot.feasibility'`.

- [ ] **Step 3: Implement the exception**

```python
# src/sparse_ot/feasibility.py
from __future__ import annotations

from dataclasses import dataclass


class InfeasibleProblemError(ValueError):
    """Raised when the sparse cost matrix's support cannot transport (a, b).

    Attributes
    ----------
    imbalance : float
        Signed mass imbalance of the worst-violating component
        (sum(a in component) - sum(b in component)).
    component_sources : list[int]
        Source-node indices in the violating component (truncated to 10).
    component_targets : list[int]
        Target-node indices in the violating component (truncated to 10).
    """

    def __init__(self, imbalance, component_sources, component_targets):
        self.imbalance = float(imbalance)
        self.component_sources = list(component_sources[:10])
        self.component_targets = list(component_targets[:10])
        msg = (
            f"sparse cost matrix support is infeasible for the given marginals: "
            f"component imbalance sum(a) - sum(b) = {self.imbalance!r}; "
            f"sources={self.component_sources!r}, "
            f"targets={self.component_targets!r}"
        )
        super().__init__(msg)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_feasibility_check.py::test_exception_attributes -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/feasibility.py tests/test_feasibility_check.py
git commit -m "feat(feasibility): InfeasibleProblemError with imbalance + component diagnostics"
```

---

## Task 3: Implement `check_feasibility`

**Files:**
- Modify: `src/sparse_ot/feasibility.py`
- Modify: `tests/test_feasibility_check.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feasibility_check.py (append)
import numpy as np
import scipy.sparse
from sparse_ot.feasibility import check_feasibility, InfeasibleProblemError


def _csr_indices(M):
    M = M.tocsr()
    return M.indptr.astype(np.int32), M.indices.astype(np.int32)


def test_connected_balanced_support_passes():
    n = 5
    M = scipy.sparse.eye(n, format='csr')  # self-edges only, balanced
    a = np.full(n, 1 / n)
    b = np.full(n, 1 / n)
    row_ptr, col_idx = _csr_indices(M)
    check_feasibility(a, b, row_ptr, col_idx)  # must not raise


def test_disconnected_balanced_support_passes():
    # Two components: {0,1} sources ↔ {0,1} targets, {2,3,4} ↔ {2,3,4}
    rows = [0, 0, 1, 1, 2, 3, 4]
    cols = [0, 1, 0, 1, 2, 3, 4]
    data = [1.0] * len(rows)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(5, 5))
    a = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
    b = np.array([0.15, 0.15, 0.1, 0.3, 0.3])  # per-component sums match
    row_ptr, col_idx = _csr_indices(M)
    check_feasibility(a, b, row_ptr, col_idx)


def test_disconnected_imbalanced_raises():
    # Component {0,1} sources have mass 0.3 but its targets only demand 0.2
    rows = [0, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4]
    data = [1.0] * 5
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(5, 5))
    a = np.array([0.2, 0.1, 0.2, 0.2, 0.3])
    b = np.array([0.1, 0.1, 0.2, 0.3, 0.3])  # comp {0}: a=0.2, b=0.1 → imbalance 0.1
    row_ptr, col_idx = _csr_indices(M)
    with pytest.raises(InfeasibleProblemError) as exc_info:
        check_feasibility(a, b, row_ptr, col_idx)
    err = exc_info.value
    assert abs(err.imbalance - 0.1) < 1e-12 or abs(err.imbalance + 0.1) < 1e-12
    assert 0 in err.component_sources
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_feasibility_check.py -v
```

Expected: 3 failures with `ImportError: cannot import name 'check_feasibility'`.

- [ ] **Step 3: Implement check_feasibility (union-find)**

```python
# src/sparse_ot/feasibility.py (append)
import numpy as np


def check_feasibility(a, b, row_ptr, col_idx, tol: float = 1e-12) -> None:
    """Raise InfeasibleProblemError if the bipartite support cannot route (a, b).

    Necessary-and-sufficient condition (unbounded edge capacities,
    sum(a) == sum(b)): every connected component of the bipartite support
    graph must have sum(a over its sources) == sum(b over its targets).

    Complexity: O(nnz · α(n+m)).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    n = a.size
    m = b.size

    # Union-find over n + m nodes: sources [0..n), targets [n..n+m).
    parent = np.arange(n + m, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            j = int(col_idx[p])
            union(i, n + j)

    # Aggregate per-component supply and demand.
    comp_supply: dict[int, float] = {}
    comp_demand: dict[int, float] = {}
    comp_sources: dict[int, list[int]] = {}
    comp_targets: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        comp_supply[r] = comp_supply.get(r, 0.0) + float(a[i])
        comp_sources.setdefault(r, []).append(i)
    for j in range(m):
        r = find(n + j)
        comp_demand[r] = comp_demand.get(r, 0.0) + float(b[j])
        comp_targets.setdefault(r, []).append(j)

    worst_root = None
    worst_imbalance = 0.0
    for r in set(comp_supply) | set(comp_demand):
        s = comp_supply.get(r, 0.0)
        d = comp_demand.get(r, 0.0)
        diff = s - d
        if abs(diff) > abs(worst_imbalance):
            worst_imbalance = diff
            worst_root = r

    if worst_root is not None and abs(worst_imbalance) > tol:
        raise InfeasibleProblemError(
            imbalance=worst_imbalance,
            component_sources=comp_sources.get(worst_root, []),
            component_targets=comp_targets.get(worst_root, []),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_feasibility_check.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/feasibility.py tests/test_feasibility_check.py
git commit -m "feat(feasibility): O(nnz) union-find check_feasibility per spec §3"
```

---

## Task 4: Wire feasibility check into `emd`

**Files:**
- Modify: `src/sparse_ot/__init__.py`
- Modify: `src/sparse_ot/emd.py:43-54`
- Test: `tests/test_emd.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emd.py (append)
import numpy as np
import scipy.sparse
import pytest
from sparse_ot import emd
from sparse_ot.feasibility import InfeasibleProblemError


def test_emd_raises_on_infeasible_sparse_support():
    # Source 0 has mass 0.5 but is only connected to target 0 which demands 0.1.
    rows = [0, 1, 1, 2, 2]
    cols = [0, 1, 2, 1, 2]
    data = [1.0] * 5
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
    a = np.array([0.5, 0.25, 0.25])
    b = np.array([0.1, 0.45, 0.45])
    with pytest.raises(InfeasibleProblemError):
        emd(a, b, M, solver='lemon')


def test_emd_skips_check_for_dense_M():
    # Dense M never raises InfeasibleProblemError regardless of marginals.
    a = np.array([0.5, 0.5])
    b = np.array([0.3, 0.7])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    emd(a, b, M)  # must not raise
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_emd.py::test_emd_raises_on_infeasible_sparse_support -v
```

Expected: FAIL — currently the call propagates a solver-level error (or returns zeros), not `InfeasibleProblemError`.

- [ ] **Step 3: Export the exception**

```python
# src/sparse_ot/__init__.py (add)
from sparse_ot.feasibility import InfeasibleProblemError  # noqa: F401
```

- [ ] **Step 4: Insert the check in emd.py**

Replace `src/sparse_ot/emd.py:43-54` with:

```python
    dense_input = not scipy.sparse.issparse(M)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, cost_sparsity_threshold)

    if (len(a), len(b)) != (n, m):
        raise ValueError(
            f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
        )

    a = a / a.sum()
    b = b / b.sum()

    selected = select_solver(n, m, nnz, solver)

    # Feasibility check (spec §2/§3): sparse-input paths only, and skipped when
    # Bonneel is explicitly selected (Bonneel solves on dense M).
    if not dense_input and selected != 'bonneel':
        from sparse_ot.feasibility import check_feasibility
        check_feasibility(a, b, row_ptr, col_idx)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_emd.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sparse_ot/__init__.py src/sparse_ot/emd.py tests/test_emd.py
git commit -m "feat(emd): raise InfeasibleProblemError before sparse solver dispatch"
```

---

## Task 5: Harden the benchmark problem generator

**Files:**
- Modify: `benchmarks/problems.py`
- Test: `tests/test_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py (append)
import numpy as np
import pytest
from benchmarks.problems import generate_knn_grid_problem
from sparse_ot.feasibility import check_feasibility


@pytest.mark.parametrize("n,k", [(50, 1), (50, 2), (200, 4), (1000, 8)])
def test_generator_produces_feasible_instance(n, k):
    a, b, M = generate_knn_grid_problem(n, k, seed=0)
    # Self-edge required at every row.
    M_csr = M.tocsr()
    for i in range(n):
        cols_i = M_csr.indices[M_csr.indptr[i]:M_csr.indptr[i + 1]]
        assert i in cols_i, f"row {i} missing self-edge"
    # Float-exact marginal balance.
    assert a.sum() == b.sum(), f"sum(a)={a.sum()!r} sum(b)={b.sum()!r}"
    # And feasibility check passes.
    row_ptr = M_csr.indptr.astype(np.int32)
    col_idx = M_csr.indices.astype(np.int32)
    check_feasibility(a, b, row_ptr, col_idx)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_benchmarks.py::test_generator_produces_feasible_instance -v
```

Expected: failures on either the self-edge assertion (for k=1 corner cases) or on `a.sum() == b.sum()` (Dirichlet draws are float-normalized but not exactly equal across two independent draws).

- [ ] **Step 3: Update the generator to guarantee the contract**

```python
# benchmarks/problems.py — replace function body
def generate_knn_grid_problem(
    n: int, k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, scipy.sparse.csr_matrix]:
    """Return (a, b, M) with a feasible-by-construction sparse support.

    Each source i connects to its k nearest target indices on the shared 1D grid;
    when k >= 1 this always includes the self-edge (i, i). Marginals are
    Dirichlet draws normalized so sum(a) == sum(b) exactly in float64.
    """
    if k < 1:
        raise ValueError("k must be >= 1 to include the self-edge")
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    # Force float-exact balance: rescale b to match a.sum() bit-for-bit.
    a = a / a.sum()
    b = b / b.sum()
    b *= a.sum() / b.sum()
    assert a.sum() == b.sum()

    half = k // 2
    rows, cols, data = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        assert lo <= i < hi, f"row {i} would miss self-edge (lo={lo}, hi={hi})"
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/problems.py tests/test_benchmarks.py
git commit -m "fix(bench): feasibility-by-construction generator; float-exact marginals"
```

---

## Task 6: Land the solver-side fix identified in Task 1

> This task's exact content depends on the root cause from Task 1. The three most likely candidates are listed below — implement the one that matches. Delete `tests/test_feasibility_root_cause.py` once the regression is covered by Task 5's tests.

- [ ] **Step 1: Re-run the reproducer to confirm it still fails on `main`**

```bash
pytest tests/test_feasibility_root_cause.py -v
```

Expected: FAIL (same error as Task 1 step 2).

- [ ] **Step 2: Apply the fix matching the Task 1 hypothesis**

**If hypothesis was "marginals not float-exact":** the generator fix in Task 5 already resolves it for the bench path. Audit `src/sparse_ot/emd.py:51-52` (the `a / a.sum()` step) — confirm it produces float-exact sums in the user path or replace with the same rescaling pattern as Task 5 step 3.

**If hypothesis was "LEMON epsilon condition still integer":** fix `src/cpp/lemon_solver.cpp` per spec §4 — replace `epsilon < 1` with `epsilon < _tolerance * _initial_max_cost`, `_tolerance = 1e-9`. Rebuild the extension:
```bash
pip install -e . --no-build-isolation
```

**If hypothesis was "OR-Tools int64 supply scaling drifts":** fix `src/sparse_ot/ortools_solver.py` — after scaling supplies to int64, redistribute the rounding residual (largest fractional remainder) so int_supply.sum() == int_demand.sum() exactly.

- [ ] **Step 3: Run the reproducer and the full bench-feasibility suite**

```bash
pytest tests/test_feasibility_root_cause.py tests/test_benchmarks.py -v
```

Expected: PASS.

- [ ] **Step 4: Delete the temporary reproducer**

```bash
git rm tests/test_feasibility_root_cause.py
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "fix(<solver>): <one-line root cause from Task 1>

Sparse band-graph instances that should be feasible were rejected as
INFEASIBLE. Root cause: <details>. Coverage moved to test_benchmarks.py."
```

---

## Task 7: Rewrite the accuracy harness (cost_ref = min over solvers)

**Files:**
- Modify: `benchmarks/bench_solvers.py`
- Test: `tests/test_benchmarks.py`

- [ ] **Step 1: Locate the accuracy code path**

```bash
grep -n "rel_cost_err\|cost_ref\|pot_reference" benchmarks/bench_solvers.py
```

Note the function name(s) that build the accuracy JSON. Record the line range.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_benchmarks.py (append)
from benchmarks.bench_solvers import compute_accuracy_cell  # adjust import to actual name


def test_cost_ref_is_min_across_feasible_solvers():
    # Mock per-solver results: two feasible, one error.
    solver_results = {
        'lemon': {'cost': 10.0, 'feasibility_a': 1e-13, 'feasibility_b': 1e-13},
        'ortools': {'cost': 10.5, 'feasibility_a': 1e-10, 'feasibility_b': 1e-10},
        'bonneel': {'error': 'skipped'},
    }
    out = compute_accuracy_cell(solver_results)
    assert out['lemon']['cost_ref'] == 10.0
    assert out['ortools']['cost_ref'] == 10.0
    assert out['lemon']['rel_cost_err'] == 0.0
    assert out['ortools']['rel_cost_err'] == pytest.approx(0.05)
    assert 'bonneel' not in out or 'error' in out['bonneel']


def test_cost_ref_excludes_infeasible_solvers():
    solver_results = {
        'lemon': {'cost': 10.0, 'feasibility_a': 1e-13, 'feasibility_b': 1e-13},
        'ortools': {'cost': 9.0,  'feasibility_a': 1e-3,  'feasibility_b': 1e-3},  # infeasible
    }
    out = compute_accuracy_cell(solver_results)
    # ortools' lower cost must NOT become the reference because its plan is not primal-feasible.
    assert out['lemon']['cost_ref'] == 10.0
    assert 'cost_ref' not in out['ortools'] or out['ortools'].get('excluded_from_reference') is True
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_benchmarks.py -v -k accuracy_cell
```

Expected: `ImportError` for `compute_accuracy_cell`, or numeric mismatches if a function of that name exists.

- [ ] **Step 4: Implement `compute_accuracy_cell`**

Add to `benchmarks/bench_solvers.py`:

```python
def compute_accuracy_cell(solver_results: dict) -> dict:
    """Compute per-solver rel_cost_err using min-across-feasible-solvers as reference.

    Solvers are considered feasible if both feasibility_a and feasibility_b
    are below 1e-8. Solvers with an 'error' key are excluded entirely.
    """
    FEAS_TOL = 1e-8
    feasible = {}
    for name, r in solver_results.items():
        if 'error' in r:
            continue
        if r.get('feasibility_a', float('inf')) > FEAS_TOL:
            continue
        if r.get('feasibility_b', float('inf')) > FEAS_TOL:
            continue
        feasible[name] = r

    if not feasible:
        return {name: dict(r) for name, r in solver_results.items()}

    cost_ref = min(r['cost'] for r in feasible.values())

    out = {}
    for name, r in solver_results.items():
        entry = dict(r)
        if name in feasible:
            entry['cost_ref'] = cost_ref
            entry['rel_cost_err'] = (r['cost'] - cost_ref) / cost_ref if cost_ref != 0 else 0.0
        elif 'error' not in entry:
            entry['excluded_from_reference'] = True
        out[name] = entry
    return out
```

Then update the existing accuracy-sweep loop to call `compute_accuracy_cell` per `(n, k)` instance and drop any code that uses POT as a reference at sparse k (POT participation is now controlled by whether it ran successfully, i.e. only at `k = n`).

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_solvers.py tests/test_benchmarks.py
git commit -m "fix(bench): cost_ref = min across feasible solvers (spec §6)"
```

---

## Task 8: Re-run the quick sweep and verify clean results

**Files:**
- Modify: `benchmarks/results/efficiency_quick.json`, `benchmarks/results/accuracy_quick.json`

- [ ] **Step 1: Re-run the quick sweep**

```bash
python benchmarks/bench_solvers.py --quick 2>&1 | tee /tmp/sweep.log
```

Expected: no `INFEASIBLE` errors for any (n, k, solver) cell where `k ≥ 1`; finite `rel_cost_err` values in the accuracy JSON.

- [ ] **Step 2: Inspect the new results**

```bash
python -c "
import json
eff = json.load(open('benchmarks/results/efficiency_quick.json'))
acc = json.load(open('benchmarks/results/accuracy_quick.json'))
for n, by_k in eff.items():
    for k, by_solver in by_k.items():
        errs = [s for s, r in by_solver.items() if 'error' in r]
        if errs:
            print(f'n={n} k={k} errors: {errs}')
print('---')
for n, by_k in acc.items():
    for k, by_solver in by_k.items():
        for s, r in by_solver.items():
            if 'rel_cost_err' in r and r['rel_cost_err'] > 1e-4:
                print(f'n={n} k={k} {s}: rel_err={r[\"rel_cost_err\"]:.2e}')
"
```

Expected: zero `errors` lines; `rel_cost_err` differences between LEMON and OR-Tools below 1e-4 on the sparse cells (their plans should largely agree).

- [ ] **Step 3: Commit the refreshed results**

```bash
git add benchmarks/results/efficiency_quick.json benchmarks/results/accuracy_quick.json
git commit -m "bench: rerun --quick after feasibility + accuracy-ref fixes"
```

---

## Task 9: Investigate Bonneel-vs-POT wall-time gap

> POT also wraps Bonneel's network_simplex_simple.h, yet `efficiency_quick.json` shows our Bonneel 30–300× slower. Diagnose before publishing routing thresholds.

**Files:**
- Inspect: `src/cpp/bonneel_solver.cpp`, `CMakeLists.txt`, `pyproject.toml`

- [ ] **Step 1: Confirm build type is Release**

```bash
python -c "
import sparse_ot._ext._bonneel as m
import os
print(os.path.realpath(m.__file__))
"
nm -a $(python -c "import sparse_ot._ext._bonneel as m; print(m.__file__)") 2>/dev/null | grep -i assert | head
```

Then check `CMakeLists.txt` for `CMAKE_BUILD_TYPE` and the wheel-build flags. Expected: `Release` with `-O3 -DNDEBUG`. If it's `Debug` or unset, that's almost certainly the gap.

- [ ] **Step 2: Profile a single dense call**

```python
# scratch
import time, numpy as np
from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd
import ot

a, b, M_sparse = generate_knn_grid_problem(1000, 1000, seed=0)
M = M_sparse.toarray()

for label, fn in [('sparse_ot', lambda: emd(a, b, M, solver='bonneel')),
                  ('pot',       lambda: ot.emd(a, b, M))]:
    t0 = time.perf_counter()
    for _ in range(3):
        fn()
    print(f"{label}: {(time.perf_counter()-t0)/3*1000:.1f} ms")
```

Note the ratio. Compare it to the JSON.

- [ ] **Step 3: Form a hypothesis and verify with one minimal fix**

Likely candidates: (a) wrapper recopies the cost matrix into a new layout per call; (b) `numItermax` default sends us deep into network-simplex iterations POT skips; (c) `-O0` build. Apply the smallest change consistent with the data; re-run Step 2.

- [ ] **Step 4: Commit fix or document the gap**

If fixed:

```bash
git add -u
git commit -m "perf(bonneel): <root cause> — <Nx> speedup on dense problems"
```

If not fixable in scope (e.g., POT uses a different solver variant), document the gap in a new section of `benchmarks/results/README.md` so the routing thresholds in Task 10 are interpreted correctly.

---

## Task 10: Derive and write routing thresholds

**Files:**
- Modify: `benchmarks/generate_report.py`
- Modify: `benchmarks/results/routing_thresholds.json`

- [ ] **Step 1: Run the full (non-quick) sweep**

```bash
python benchmarks/bench_solvers.py 2>&1 | tee /tmp/full_sweep.log
```

Expected: completes without errors; produces `efficiency.json` and `accuracy.json`.

- [ ] **Step 2: Write the failing test for threshold derivation**

```python
# tests/test_benchmarks.py (append)
def test_derive_thresholds_picks_crossover():
    from benchmarks.generate_report import derive_thresholds
    # Synthetic efficiency data: bonneel wins at k≥64, lemon below; lemon wins
    # at n≤500000, ortools above.
    eff = {
        '1000': {
            '4':  {'bonneel': {'wall_time_s': 2.0}, 'lemon': {'wall_time_s': 0.5}, 'ortools': {'wall_time_s': 1.0}},
            '64': {'bonneel': {'wall_time_s': 0.4}, 'lemon': {'wall_time_s': 0.5}, 'ortools': {'wall_time_s': 0.7}},
        },
        '1000000': {
            '4':  {'bonneel': {'wall_time_s': None}, 'lemon': {'wall_time_s': 60.0}, 'ortools': {'wall_time_s': 30.0}},
            '64': {'bonneel': {'wall_time_s': None}, 'lemon': {'wall_time_s': 40.0}, 'ortools': {'wall_time_s': 50.0}},
        },
    }
    out = derive_thresholds(eff)
    assert 4 <= out['bonneel_lemon'] <= 64
    assert 1000 <= out['lemon_ortools'] <= 1000000
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_benchmarks.py::test_derive_thresholds_picks_crossover -v
```

Expected: `ImportError` for `derive_thresholds`.

- [ ] **Step 4: Implement `derive_thresholds` in generate_report.py**

```python
# benchmarks/generate_report.py (append)
def derive_thresholds(eff: dict) -> dict:
    """Derive routing thresholds from an efficiency sweep.

    bonneel_lemon: the smallest k at which bonneel beats lemon for any n.
    lemon_ortools: the smallest n at which ortools beats lemon for the lowest
                   k where both ran successfully.
    """
    bonneel_lemon = None
    for n_str, by_k in eff.items():
        for k_str, by_s in sorted(by_k.items(), key=lambda kv: int(kv[0])):
            wb = by_s.get('bonneel', {}).get('wall_time_s')
            wl = by_s.get('lemon', {}).get('wall_time_s')
            if wb is not None and wl is not None and wb < wl:
                k = int(k_str)
                bonneel_lemon = k if bonneel_lemon is None else min(bonneel_lemon, k)
                break

    lemon_ortools = None
    for n_str in sorted(eff, key=int):
        by_k = eff[n_str]
        for k_str, by_s in by_k.items():
            wl = by_s.get('lemon', {}).get('wall_time_s')
            wo = by_s.get('ortools', {}).get('wall_time_s')
            if wl is not None and wo is not None and wo < wl:
                n = int(n_str)
                lemon_ortools = n if lemon_ortools is None else min(lemon_ortools, n)
                break

    return {
        'bonneel_lemon': bonneel_lemon if bonneel_lemon is not None else 128,
        'lemon_ortools': lemon_ortools if lemon_ortools is not None else 1_000_000,
    }
```

Add a `__main__` step that loads `efficiency.json`, calls `derive_thresholds`, and writes `routing_thresholds.json`.

- [ ] **Step 5: Run test and produce the new thresholds file**

```bash
pytest tests/test_benchmarks.py::test_derive_thresholds_picks_crossover -v
python benchmarks/generate_report.py
cat benchmarks/results/routing_thresholds.json
```

Expected: test PASS; `routing_thresholds.json` reflects empirical crossover (not the default 128 / 1_000_000 unless the data genuinely supports those).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/generate_report.py benchmarks/results/routing_thresholds.json benchmarks/results/efficiency.json benchmarks/results/accuracy.json tests/test_benchmarks.py
git commit -m "feat(bench): derive routing thresholds from empirical crossover"
```

---

## Self-Review

- **Spec coverage:** §2 (InfeasibleProblemError contract) → Task 2 + Task 4. §3 (union-find feasibility check in data flow) → Task 3 + Task 4. §6 (feasibility-by-construction generator) → Task 5. §6 (cost_ref = min-of-solvers accuracy) → Task 7. §7 (test_feasibility_check.py row) → Tasks 2, 3, 4. Routing thresholds → Task 10.
- **Implementation gaps from current code state:** sparse solvers returning INFEASIBLE on feasible problems → Task 1 + Task 6. Bonneel-vs-POT 30–300× slowdown → Task 9. Default routing thresholds never updated → Task 10.
- **Placeholders:** none — every step shows real code or real commands.
- **Type consistency:** `InfeasibleProblemError` attributes (`imbalance`, `component_sources`, `component_targets`) match between Task 2 definition, Task 3 raises, Task 4 catches, and the spec.
- **Risks:** Task 1's exact fix is hypothesis-driven; Task 6 has three branches and the wrong one will not resolve the reproducer — re-run Task 1 if so. Task 9 may not be fixable in scope; the plan permits documenting rather than fixing.
