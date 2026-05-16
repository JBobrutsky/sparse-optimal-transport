# Plan 2: Sparse Utilities + LEMON Solver + Routing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend sparse-ot so dense and scipy-sparse cost matrices route automatically to Bonneel (near-dense) or LEMON CostScaling (sparse), passing all correctness tests against POT.

**Architecture:** `sparse_utils.py` converts any cost matrix (dense numpy or scipy sparse) to CSR int32/float64; `routing.py` picks the solver using k = nnz/n vs a JSON threshold table; `lemon_solver.cpp` wraps LEMON's CostScaling (vendored, float64-patched via custom traits + termination fix) via pybind11 and returns COO triplets; `emd.py` orchestrates these pieces and returns dense numpy for dense input or scipy CSR for sparse input.

**Tech Stack:** Python 3.12 (uv), scipy 1.10+, LEMON 1.3.1 (vendored headers), pybind11, CMake 3.18+, POT (tests only), pytest

---

## File Map

| File | Role |
|---|---|
| `src/sparse_ot/sparse_utils.py` | `to_csr()`: dense+scipy→CSR int32/float64 with threshold |
| `src/sparse_ot/routing.py` | `select_solver()`: dispatch via k and problem size |
| `benchmarks/results/routing_thresholds.json` | Conservative defaults: bonneel_lemon=128, lemon_ortools=1M |
| `src/cpp/lemon/` | Vendored LEMON 1.3.1 headers (CostScaling subset, ~15 files) |
| `src/cpp/lemon/VENDORING.md` | Source + license record |
| `src/cpp/lemon/PATCHES.md` | Float64 patch documentation |
| `src/cpp/lemon_solver.cpp` | pybind11: CSR → LEMON CostScaling → COO float64 |
| `CMakeLists.txt` | Add `_lemon` pybind11 extension target |
| `src/sparse_ot/emd.py` | Add routing, LEMON path, scipy sparse input support |
| `tests/test_sparse_utils.py` | CSR conversion correctness tests |
| `tests/test_routing.py` | Solver selection logic tests |
| `tests/test_lemon_accuracy.py` | Float64 patch + accuracy vs POT sweep |
| `tests/test_emd.py` | Add LEMON path + scipy sparse input tests |

---

## Task 1: sparse_utils.py (TDD)

**Files:**
- Create: `tests/test_sparse_utils.py`
- Create: `src/sparse_ot/sparse_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sparse_utils.py
import numpy as np
import scipy.sparse
import pytest
from sparse_ot.sparse_utils import to_csr


def test_dense_all_edges():
    M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert n == 3 and m == 2 and nnz == 6
    assert row_ptr.dtype == np.int32
    assert col_idx.dtype == np.int32
    assert costs.dtype == np.float64
    assert len(row_ptr) == n + 1
    assert len(col_idx) == nnz
    assert len(costs) == nnz


def test_dense_threshold_removes_edges():
    # |0.0| <= 0.5 and |0.5| <= 0.5 are excluded; |1.0|, |2.0| remain
    M = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, cost_sparsity_threshold=0.5)
    assert nnz == 2
    assert set(zip(col_idx.tolist(), costs.tolist())) == {(1, 1.0), (1, 2.0)}


def test_dense_exact_zeros_excluded_by_default():
    # Exact zero at (0,0) must be excluded even with threshold=0.0
    M = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert nnz == 3
    # row_ptr: [0, 1, 3] — row 0 has 1 edge, row 1 has 2 edges
    assert row_ptr[0] == 0
    assert row_ptr[1] == 1
    assert row_ptr[2] == 3


def test_dense_fully_dense():
    M = np.ones((4, 5))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert n == 4 and m == 5 and nnz == 20


def test_dense_single_edge():
    M = np.zeros((3, 3))
    M[1, 2] = 5.0
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert nnz == 1
    assert costs[0] == 5.0


def test_scipy_csr_roundtrip():
    data = np.array([1.0, 2.0, 3.0])
    row = np.array([0, 1, 1])
    col = np.array([0, 0, 1])
    M_sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp)
    assert n == 2 and m == 2 and nnz == 3
    assert row_ptr.dtype == np.int32
    assert col_idx.dtype == np.int32
    assert costs.dtype == np.float64


def test_scipy_coo_converted():
    data = np.array([1.0, 2.0])
    row = np.array([0, 1])
    col = np.array([1, 0])
    M_sp = scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp)
    assert nnz == 2


def test_scipy_sparse_ignores_threshold():
    # threshold is only applied to dense inputs; scipy sparse is taken as-is
    data = np.array([0.1, 0.5, 1.0])
    row = np.array([0, 0, 1])
    col = np.array([0, 1, 0])
    M_sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp, cost_sparsity_threshold=0.5)
    # scipy path does NOT apply the threshold; nnz matches the sparse matrix's stored nnz
    assert nnz == 3
```

- [ ] **Step 2: Run tests — confirm ImportError**

```bash
pytest tests/test_sparse_utils.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'to_csr' from 'sparse_ot.sparse_utils'`

- [ ] **Step 3: Implement sparse_utils.py**

```python
# src/sparse_ot/sparse_utils.py
import numpy as np
import scipy.sparse


def to_csr(M, cost_sparsity_threshold=0.0):
    """Convert a cost matrix to CSR format for the C++ solvers.

    For scipy sparse input, converts to CSR and casts indices/data; threshold
    is NOT applied (the caller controls sparsity via the scipy matrix itself).
    For dense numpy input, entries with |M[i,j]| <= threshold are dropped.
    Exact zeros are always dropped from dense input.

    Returns
    -------
    row_ptr : int32 ndarray, shape (n+1,)
    col_idx : int32 ndarray, shape (nnz,)
    costs   : float64 ndarray, shape (nnz,)
    n, m    : int — source and target sizes
    nnz     : int — number of edges
    """
    if scipy.sparse.issparse(M):
        csr = M.tocsr().astype(np.float64)
        csr.eliminate_zeros()
        n, m = csr.shape
        return (
            csr.indptr.astype(np.int32),
            csr.indices.astype(np.int32),
            np.asarray(csr.data, dtype=np.float64),
            n, m, int(csr.nnz),
        )

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    n, m = M.shape

    mask = np.abs(M) > cost_sparsity_threshold  # excludes exact zeros and <= threshold

    row_counts = mask.sum(axis=1).astype(np.int32)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(row_counts, out=row_ptr[1:])
    nnz = int(row_ptr[n])

    col_idx = np.empty(nnz, dtype=np.int32)
    costs = np.empty(nnz, dtype=np.float64)
    for i in range(n):
        s, e = int(row_ptr[i]), int(row_ptr[i + 1])
        js = np.where(mask[i])[0].astype(np.int32)
        col_idx[s:e] = js
        costs[s:e] = M[i, js]

    return row_ptr, col_idx, costs, n, m, nnz
```

- [ ] **Step 4: Run tests — confirm 8 passed**

```bash
pytest tests/test_sparse_utils.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sparse_utils.py src/sparse_ot/sparse_utils.py
git commit -m "feat: sparse_utils.to_csr() — dense and scipy sparse to CSR int32/float64"
```

---

## Task 2: routing.py + thresholds JSON (TDD)

**Files:**
- Create: `benchmarks/results/routing_thresholds.json`
- Create: `src/sparse_ot/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Create benchmarks directory and write default thresholds**

```bash
mkdir -p benchmarks/results
```

Write `benchmarks/results/routing_thresholds.json`:
```json
{
  "bonneel_lemon": 128,
  "lemon_ortools": 1000000
}
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_routing.py
import pytest
from sparse_ot.routing import select_solver


def test_high_k_routes_to_bonneel():
    # k = nnz/n = 2000/10 = 200 > 128 → bonneel
    assert select_solver(n=10, m=10, nnz=2000) == 'bonneel'


def test_low_k_small_n_routes_to_lemon():
    # k = 100/10 = 10 < 128, n=10 < 1_000_000 → lemon
    assert select_solver(n=10, m=10, nnz=100) == 'lemon'


def test_large_n_routes_to_ortools():
    # k = 4_000_000/2_000_000 = 2 < 128, n=2_000_000 > 1_000_000 → ortools
    assert select_solver(n=2_000_000, m=2_000_000, nnz=4_000_000) == 'ortools'


def test_boundary_n_just_below_lemon():
    # k=10 < 128, n=999_999 < 1_000_000 → lemon
    assert select_solver(n=999_999, m=10, nnz=999_999 * 10) == 'lemon'


def test_boundary_n_just_above_ortools():
    # k=10 < 128, n=1_000_001 > 1_000_000 → ortools
    assert select_solver(n=1_000_001, m=10, nnz=1_000_001 * 10) == 'ortools'


def test_override_bonneel_beats_lemon_routing():
    # k=10 would normally → lemon, but override wins
    assert select_solver(n=10, m=10, nnz=100, solver='bonneel') == 'bonneel'


def test_override_lemon_beats_bonneel_routing():
    # k=200 would normally → bonneel, but override wins
    assert select_solver(n=10, m=10, nnz=2000, solver='lemon') == 'lemon'


def test_override_ortools():
    assert select_solver(n=10, m=10, nnz=100, solver='ortools') == 'ortools'


def test_large_k_fully_dense_routes_to_bonneel():
    # n=200, k=200 > 128 → bonneel
    assert select_solver(n=200, m=200, nnz=200 * 200) == 'bonneel'


def test_small_n_fully_dense_routes_to_lemon():
    # n=10, k=10 < 128 → lemon
    assert select_solver(n=10, m=10, nnz=10 * 10) == 'lemon'
```

- [ ] **Step 3: Run tests — confirm ImportError**

```bash
pytest tests/test_routing.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'select_solver' from 'sparse_ot.routing'`

- [ ] **Step 4: Implement routing.py**

```python
# src/sparse_ot/routing.py
import json
from pathlib import Path

_THRESHOLDS_PATH = (
    Path(__file__).parent.parent.parent / "benchmarks" / "results" / "routing_thresholds.json"
)
_DEFAULTS = {"bonneel_lemon": 128, "lemon_ortools": 1_000_000}


def _load_thresholds():
    try:
        return json.loads(_THRESHOLDS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _DEFAULTS


def select_solver(n: int, m: int, nnz: int, solver: str | None = None) -> str:
    """Select the best solver for an OT problem.

    Parameters
    ----------
    n, m    : int   Source and target sizes.
    nnz     : int   Number of edges in the cost graph.
    solver  : str or None   Override ('bonneel', 'lemon', 'ortools') or None for auto.

    Returns
    -------
    'bonneel', 'lemon', or 'ortools'
    """
    if solver is not None:
        return solver
    thresholds = _load_thresholds()
    k = nnz / n if n > 0 else 0.0
    if k > thresholds["bonneel_lemon"]:
        return "bonneel"
    if n > thresholds["lemon_ortools"]:
        return "ortools"
    return "lemon"
```

- [ ] **Step 5: Run tests — confirm 10 passed**

```bash
pytest tests/test_routing.py -v
```
Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add benchmarks/results/routing_thresholds.json src/sparse_ot/routing.py tests/test_routing.py
git commit -m "feat: routing.py + routing_thresholds.json with conservative defaults"
```

---

## Task 3: Vendor LEMON Headers

**Files:**
- Create: `src/cpp/lemon/` (directory with headers)
- Create: `src/cpp/lemon/VENDORING.md`

- [ ] **Step 1: Download LEMON 1.3.1 and extract**

```bash
curl -L -o /tmp/lemon-1.3.1.tar.gz \
  http://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
tar -xzf /tmp/lemon-1.3.1.tar.gz -C /tmp/
ls /tmp/lemon-1.3.1/lemon/
```
Expected: a directory of `.h` files plus a `bits/` subdirectory.

- [ ] **Step 2: Extract CostScaling dependency tree**

```bash
mkdir -p src/cpp/lemon/bits

# Top-level headers CostScaling needs
for f in cost_scaling.h core.h list_graph.h maps.h tolerance.h math.h error.h; do
    cp /tmp/lemon-1.3.1/lemon/$f src/cpp/lemon/
done

# config.h (may or may not exist; skip if absent)
cp /tmp/lemon-1.3.1/lemon/config.h src/cpp/lemon/ 2>/dev/null || true

# bits/ subdirectory
for f in enable_if.h traits.h array_map.h map_extender.h default_map.h \
          graph_extender.h vector_map.h; do
    cp /tmp/lemon-1.3.1/lemon/bits/$f src/cpp/lemon/bits/
done
```

- [ ] **Step 3: Verify compilation finds all headers**

```bash
cat > /tmp/lemon_probe.cpp << 'EOF'
#include "lemon/core.h"
#include "lemon/list_graph.h"
#include "lemon/cost_scaling.h"
int main() { return 0; }
EOF
g++ -std=c++17 -I src/cpp /tmp/lemon_probe.cpp -o /tmp/lemon_probe 2>&1
```

Expected: clean compile with no errors. If you see `fatal error: lemon/bits/<something>.h: No such file or directory`, copy that file from `/tmp/lemon-1.3.1/lemon/bits/` and re-run. Repeat until the compile succeeds.

- [ ] **Step 4: Write VENDORING.md**

```markdown
# LEMON CostScaling Headers

Source: http://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz  
Version: LEMON 1.3.1  
License: Boost Software License 1.0 (see individual file headers)  
Vendored: 2026-05-16  
Subset: CostScaling and its dependency tree (~15 headers)  
Modifications: see PATCHES.md
```

- [ ] **Step 5: Commit vendored headers**

```bash
git add src/cpp/lemon/
git commit -m "vendor: LEMON 1.3.1 CostScaling headers (unpatched)"
```

---

## Task 4: Apply Float64 Patch

**Files:**
- Modify: `src/cpp/lemon/cost_scaling.h`
- Create: `src/cpp/lemon/PATCHES.md`

The problem: LEMON's epsilon-scaling loop terminates when `_epsilon >= 1`. With integer costs this is fine (epsilon scales from max_cost down to 1). With float64 costs in [0, 1], max_cost ≈ 1.0 and the condition is immediately false after the first epsilon reduction — producing a one-iteration solve.

The fix: replace `_epsilon >= 1` with `_epsilon >= _tolerance * _initial_max_cost` so termination is relative to the initial cost scale.

- [ ] **Step 1: Locate the epsilon fields and the loop condition**

```bash
grep -n "_epsilon\|epsilon" src/cpp/lemon/cost_scaling.h | head -40
```

Find:
1. The line that declares `LargeCost _epsilon;` in the private data section (near other `_` fields).
2. The line(s) in `_init()` or `init()` where `_epsilon` is first assigned (typically `_epsilon = _max_cost;` or `_epsilon = _max_cost / _alpha;`).
3. The outer loop condition containing `_epsilon >= 1` (in the `run()`, `_startAugment()`, or `_startPush()` method).

```bash
grep -n ">= 1\|>= _alpha\|epsilon.*1\b" src/cpp/lemon/cost_scaling.h
```

The outer loop will look like one of:
```cpp
for (_epsilon = _max_cost; _epsilon >= 1; ...
```
or a while loop:
```cpp
while (_epsilon >= 1) {
```

- [ ] **Step 2: Add `_tolerance` and `_initial_max_cost` fields**

Find the private data block containing `LargeCost _epsilon;`. Add two lines directly after it:

```cpp
// Before (find this exact declaration):
LargeCost _epsilon;

// After (replace with):
LargeCost _epsilon;
LargeCost _initial_max_cost;
double _tolerance;
```

- [ ] **Step 3: Initialize the new fields at the point where `_epsilon` is first set**

Find the line that sets `_epsilon` for the first time (it will look like `_epsilon = _max_cost;` or similar). Add the two initializations directly after it:

```cpp
// Before (find this):
_epsilon = _max_cost;

// After (replace with):
_epsilon = _max_cost;
_initial_max_cost = _max_cost;
_tolerance = 1e-9;
```

If the initial assignment is `_epsilon = _max_cost / _alpha;`, still add the two lines after it — `_initial_max_cost` captures `_max_cost`, not `_epsilon`.

Verify:
```bash
grep -n "_initial_max_cost\|_tolerance" src/cpp/lemon/cost_scaling.h
```
Expected: 3 lines (declaration + 2 initializations).

- [ ] **Step 4: Replace the loop termination condition**

The outer loop checks `_epsilon >= 1`. Replace `1` with `_tolerance * _initial_max_cost`:

If the loop is:
```cpp
for (_epsilon = _max_cost; _epsilon >= 1;
```
Change to:
```cpp
for (_epsilon = _max_cost; _epsilon >= _tolerance * _initial_max_cost;
```

If the loop uses a `while`:
```cpp
while (_epsilon >= 1) {
```
Change to:
```cpp
while (_epsilon >= _tolerance * _initial_max_cost) {
```

Verify you changed exactly one occurrence:
```bash
grep -n "_tolerance \* _initial_max_cost" src/cpp/lemon/cost_scaling.h
```
Expected: 1 line.

- [ ] **Step 5: Write PATCHES.md**

```markdown
# Patches to LEMON CostScaling

## Float64 termination criterion (2026-05-16)

**File:** `cost_scaling.h`

**Problem:** The epsilon-scaling loop terminates when `_epsilon >= 1`, which assumes
integer costs. With float64 costs in [0, 1], `_epsilon` starts at ~1 and drops below 1
after one step, giving a one-iteration solve with poor solution quality.

**Changes:**
1. Added `LargeCost _initial_max_cost;` and `double _tolerance;` to the private data section.
2. `_initial_max_cost = _max_cost; _tolerance = 1e-9;` added at epsilon initialization.
3. Loop condition `_epsilon >= 1` replaced with `_epsilon >= _tolerance * _initial_max_cost`.

**Effect:** The algorithm runs until epsilon is 1e-9 times the initial max cost, providing
the same relative precision for any cost scale.
```

- [ ] **Step 6: Commit patch**

```bash
git add src/cpp/lemon/cost_scaling.h src/cpp/lemon/PATCHES.md
git commit -m "fix: LEMON float64 patch — replace epsilon>=1 with relative tolerance 1e-9"
```

---

## Task 5: lemon_solver.cpp + Build

**Files:**
- Create: `src/cpp/lemon_solver.cpp`
- Modify: `CMakeLists.txt`

The design: a custom traits struct `Float64Traits<GR>` sets `Value = Cost = LargeCost = double`, bypassing LEMON's default `LargeCost = long long`. We build a bipartite `ListDigraph` from CSR input (n source nodes, m sink nodes, nnz arcs), attach supply/demand and costs via `NodeMap`/`ArcMap`, run `CostScaling::run(OPTIMAL)`, then extract non-zero flows using per-arc `ArcMap` lookups (correct regardless of `ArcIt` iteration order).

- [ ] **Step 1: Write lemon_solver.cpp**

```cpp
// src/cpp/lemon_solver.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "lemon/core.h"
#include "lemon/list_graph.h"

namespace py = pybind11;
using namespace lemon;

typedef ListDigraph Graph;
typedef Graph::Node Node;
typedef Graph::Arc  Arc;

// Custom traits: all numeric types are double, so epsilon arithmetic
// stays in double rather than long long. Matches the float64 patch.
template <typename GR>
struct Float64Traits {
    typedef GR     Digraph;
    typedef double Value;      // supply / flow type
    typedef double Cost;       // edge cost type
    typedef double LargeCost;  // internal large-cost type
};

#include "lemon/cost_scaling.h"

typedef CostScaling<Graph, double, double, Float64Traits<Graph>> CS;

// Solve a sparse balanced OT problem using LEMON CostScaling (float64-patched).
//
// a        : source distribution (n,) float64, sums to 1.0
// b        : target distribution (m,) float64, sums to 1.0
// row_ptr  : CSR row pointer (n+1,) int32
// col_idx  : CSR column indices (nnz,) int32
// costs    : CSR edge costs (nnz,) float64
// numItermax: iteration limit passed to LEMON
//
// Returns COO triplets (row_indices, col_indices, values) for arcs with flow > 0.
std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>>
solve_sparse(
    py::array_t<double,  py::array::c_style | py::array::forcecast> a,
    py::array_t<double,  py::array::c_style | py::array::forcecast> b,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> col_idx_arr,
    py::array_t<double,  py::array::c_style | py::array::forcecast> costs_arr,
    int numItermax
) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());

    const double*   ap = a.data();
    const double*   bp = b.data();
    const int32_t*  rp = row_ptr_arr.data();
    const int32_t*  ci = col_idx_arr.data();
    const double*   cp = costs_arr.data();

    // Build bipartite graph: nodes 0..n-1 are sources, nodes n..n+m-1 are sinks.
    Graph g;
    std::vector<Node> nodes;
    nodes.reserve(n + m);
    for (int k = 0; k < n + m; ++k)
        nodes.push_back(g.addNode());

    Graph::NodeMap<double>  supply(g);
    for (int i = 0; i < n; ++i)  supply[nodes[i]]     =  ap[i];
    for (int j = 0; j < m; ++j)  supply[nodes[n + j]] = -bp[j];

    Graph::ArcMap<double>   cost_map(g);
    // Store (source_i, target_j) per arc so we can reconstruct COO regardless
    // of the order in which ArcIt visits arcs.
    Graph::ArcMap<int32_t>  arc_src(g), arc_dst(g);

    for (int i = 0; i < n; ++i) {
        for (int32_t ptr = rp[i]; ptr < rp[i + 1]; ++ptr) {
            int j   = ci[ptr];
            Arc arc = g.addArc(nodes[i], nodes[n + j]);
            cost_map[arc] = cp[ptr];
            arc_src[arc]  = i;
            arc_dst[arc]  = j;
        }
    }

    CS cs(g);
    cs.supplyMap(supply);
    cs.costMap(cost_map);
    cs.run(CS::OPTIMAL, numItermax);

    // Collect non-zero flows as COO.
    std::vector<int32_t> out_rows, out_cols;
    std::vector<double>  out_vals;

    for (Graph::ArcIt arc(g); arc != INVALID; ++arc) {
        double flow = cs.flow(arc);
        if (flow > 0.0) {
            out_rows.push_back(arc_src[arc]);
            out_cols.push_back(arc_dst[arc]);
            out_vals.push_back(flow);
        }
    }

    const ssize_t sz = static_cast<ssize_t>(out_rows.size());
    py::array_t<int32_t> rows_out(sz), cols_out(sz);
    py::array_t<double>  vals_out(sz);
    std::copy(out_rows.begin(), out_rows.end(), rows_out.mutable_data());
    std::copy(out_cols.begin(), out_cols.end(), cols_out.mutable_data());
    std::copy(out_vals.begin(), out_vals.end(), vals_out.mutable_data());

    return {rows_out, cols_out, vals_out};
}

PYBIND11_MODULE(_lemon, m) {
    m.doc() = "LEMON CostScaling for sparse balanced OT (float64-patched)";
    m.def(
        "solve_sparse", &solve_sparse,
        py::arg("a"), py::arg("b"),
        py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
        py::arg("numItermax") = 100000,
        "Solve sparse balanced OT. Returns COO (row_indices, col_indices, values)."
    );
}
```

- [ ] **Step 2: Add `_lemon` target to CMakeLists.txt**

In `CMakeLists.txt`, add the following block directly after the existing `install(TARGETS _bonneel ...)` line:

```cmake
pybind11_add_module(_lemon src/cpp/lemon_solver.cpp)
target_include_directories(_lemon PRIVATE src/cpp)
target_compile_options(_lemon PRIVATE -O3)
install(TARGETS _lemon DESTINATION sparse_ot/_ext)
```

- [ ] **Step 3: Build both extensions**

```bash
uv pip install --no-build-isolation -e . 2>&1 | tail -20
```
Expected: build output ending in `Successfully installed sparse-ot-0.1.0` with both `_bonneel` and `_lemon` compiled.

**If you see a compilation error** about a missing method on `CS` (e.g., `supplyMap` or `costMap` not found):
```bash
grep -n "supplyMap\|costMap\|setSupply\|setCost" src/cpp/lemon/cost_scaling.h | head -20
```
LEMON 1.3.1 exposes these as `supplyMap(map)` and `costMap(map)`. If the method name differs, update the calls in `lemon_solver.cpp`.

**If you see a compilation error** about `Float64Traits` not matching the expected traits interface:
```bash
grep -n "typename TR\|TR::Value\|TR::Cost\|TR::LargeCost" src/cpp/lemon/cost_scaling.h | head -10
```
Verify `Float64Traits` provides all the typedefs that `CostScaling` expects from its `TR` parameter.

**If you see a compilation error** about `_initial_max_cost` or `_tolerance` undeclared:
- Double-check Task 4 Step 2 was applied to the correct file: `src/cpp/lemon/cost_scaling.h`
- Run `grep -n "_initial_max_cost" src/cpp/lemon/cost_scaling.h` — should show 2+ lines.

- [ ] **Step 4: Smoke-test the LEMON extension directly**

```bash
python - <<'EOF'
import numpy as np
from sparse_ot._ext import _lemon

# 2-source, 2-sink balanced problem
# Source 0 (supply 0.5) connected to both sinks at cost [0, 1]
# Source 1 (supply 0.5) connected to both sinks at cost [1, 0]
# Optimal: route source 0 → sink 0, source 1 → sink 1
a = np.array([0.5, 0.5])
b = np.array([0.5, 0.5])
row_ptr = np.array([0, 2, 4], dtype=np.int32)
col_idx = np.array([0, 1, 0, 1], dtype=np.int32)
costs   = np.array([0.0, 1.0, 1.0, 0.0])

rows, cols, vals = _lemon.solve_sparse(a, b, row_ptr, col_idx, costs, 100000)
print("rows:", rows)   # [0, 1]
print("cols:", cols)   # [0, 1]
print("vals:", vals)   # [0.5, 0.5]
print("total cost:", sum(c * v for c, v in zip(
    [costs[r * 2 + c] for r, c in zip(rows, cols)], vals)))  # 0.0
EOF
```
Expected: rows=[0,1], cols=[0,1], vals=[0.5, 0.5], total cost=0.0.

- [ ] **Step 5: Commit**

```bash
git add src/cpp/lemon_solver.cpp CMakeLists.txt
git commit -m "feat: lemon_solver.cpp pybind11 wrapper — CSR → CostScaling → COO float64"
```

---

## Task 6: test_lemon_accuracy.py

**Files:**
- Create: `tests/test_lemon_accuracy.py`

- [ ] **Step 1: Write accuracy tests**

```python
# tests/test_lemon_accuracy.py
import numpy as np
import pytest
import scipy.sparse
import ot

from sparse_ot import emd, emd2


def _dense_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0, 1, (n, n))
    return a, b, M


@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1.0, 1e1, 1e3])
def test_lemon_cost_vs_pot_various_scales(scale):
    """LEMON relative cost error < 1e-6 across five orders of cost magnitude."""
    a, b, M = _dense_problem(20)
    M_scaled = M * scale
    cost_lemon = emd2(a, b, M_scaled, solver='lemon')
    cost_pot   = ot.emd2(a, b, M_scaled)
    rel_err = abs(cost_lemon - cost_pot) / max(abs(cost_pot), 1e-15)
    assert rel_err < 1e-6, f"scale={scale}: lemon={cost_lemon}, pot={cost_pot}, rel_err={rel_err}"


@pytest.mark.parametrize("n", [5, 20, 50])
def test_lemon_marginals_dense_input(n):
    """Transport plan from LEMON has correct row and column marginals."""
    a, b, M = _dense_problem(n)
    G = emd(a, b, M, solver='lemon')
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)


def test_lemon_plan_vs_pot_dense(seed=7):
    """LEMON transport plan matches POT's dense plan to atol=1e-6."""
    a, b, M = _dense_problem(15, seed=seed)
    G_lemon = emd(a, b, M, solver='lemon')
    G_pot   = ot.emd(a, b, M)
    np.testing.assert_allclose(G_lemon, G_pot, atol=1e-6)


def test_lemon_sparse_input_returns_csr():
    """scipy sparse cost matrix → LEMON → scipy CSR output."""
    rng = np.random.default_rng(0)
    n = 10
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    rows_idx = [i for i in range(n) for _ in range(3)]
    cols_idx = [(i + k) % n for i in range(n) for k in range(3)]
    data = rng.uniform(0.1, 1.0, len(rows_idx))
    M_sp = scipy.sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))
    G = emd(a, b, M_sp, solver='lemon')
    assert scipy.sparse.issparse(G)
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=1)).ravel(), a, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=0)).ravel(), b, atol=1e-9
    )


def test_lemon_sparse_cost_matches_pot():
    """LEMON on a sparse problem matches POT's solution cost (via dense fallback)."""
    rng = np.random.default_rng(3)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    # Fully connected (dense) sparse matrix so we can compare to POT
    rows_idx = [i for i in range(n) for j in range(n)]
    cols_idx = [j for i in range(n) for j in range(n)]
    data = rng.uniform(0, 1, n * n)
    M_dense = np.array(data).reshape(n, n)
    M_sp = scipy.sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))
    cost_lemon = emd2(a, b, M_sp, solver='lemon')
    cost_pot   = ot.emd2(a, b, M_dense)
    rel_err = abs(cost_lemon - cost_pot) / max(abs(cost_pot), 1e-15)
    assert rel_err < 1e-6, f"rel_err={rel_err}"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_lemon_accuracy.py -v
```
Expected: all tests pass. If `test_lemon_cost_vs_pot_various_scales` fails for any scale, re-check the LEMON patch in Task 4: the loop condition replacement must use `_tolerance * _initial_max_cost`, and the initialization of `_initial_max_cost` must happen at the right place. Debug by printing the cost values and checking they converge.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lemon_accuracy.py
git commit -m "test: LEMON float64 accuracy tests — cost and marginal correctness vs POT"
```

---

## Task 7: Update emd.py + test_emd.py

**Files:**
- Modify: `src/sparse_ot/emd.py`
- Modify: `tests/test_emd.py`

The routing rules mean most small dense problems (k = n ≤ 128) now go through LEMON. Since `emd()` returns `G.toarray()` (dense numpy) for dense input, all existing tests remain valid without modification. The updates add LEMON-explicit tests and scipy sparse input tests.

- [ ] **Step 1: Rewrite emd.py**

```python
# src/sparse_ot/emd.py
import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.routing import select_solver


def emd(a, b, M, numItermax=100000, log=False, center_dual=True,
        cost_sparsity_threshold=0.0, solver=None):
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
    if solver == 'ortools':
        raise NotImplementedError(
            "'ortools' solver is not yet implemented. "
            "Use None, 'bonneel', or 'lemon'."
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

    # LEMON path
    from sparse_ot._ext import _lemon
    rows, cols, vals = _lemon.solve_sparse(
        a, b, row_ptr, col_idx, costs, numItermax
    )
    G_sp = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    G = G_sp.toarray() if dense_input else G_sp

    if log:
        return G, {}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
         cost_sparsity_threshold=0.0, solver=None):
    """OT cost between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd2(). Returns a float scalar.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like (n, m) or scipy sparse (n, m)
    numItermax : int
    log : bool
    return_matrix : bool — if True, also return the transport plan G
    cost_sparsity_threshold : float
    solver : str or None

    Returns
    -------
    cost : float
    G : ndarray or scipy CSR — only if return_matrix=True
    """
    G = emd(a, b, M, numItermax=numItermax, log=False,
            cost_sparsity_threshold=cost_sparsity_threshold, solver=solver)

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

- [ ] **Step 2: Run the existing test suite — all 12 tests must still pass**

```bash
pytest tests/test_emd.py -v
```
Expected: `12 passed`.

If `test_emd_matches_pot_rectangular` or `test_emd_matches_pot_square` fails — those use n,m ≤ 20, so k ≤ 20 → LEMON path. The LEMON path returns `G.toarray()` (dense numpy), so `np.testing.assert_allclose(G, G_ref)` should work. If it fails, check the accuracy tolerance or whether `_lemon` was rebuilt after the LEMON patch.

- [ ] **Step 3: Add LEMON-specific and sparse-input tests to test_emd.py**

Append to the end of `tests/test_emd.py`:

```python
# --- LEMON path tests ---

def test_emd_lemon_override():
    a, b, M = _problem(8, 8)
    G = sparse_ot.emd(a, b, M, solver='lemon')
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd2_lemon_override():
    a, b, M = _problem(10, 10)
    cost = sparse_ot.emd2(a, b, M, solver='lemon')
    cost_ref = ot.emd2(a, b, M)
    assert abs(cost - cost_ref) / abs(cost_ref) < 1e-6


def test_emd_scipy_sparse_input():
    """scipy CSR cost matrix is accepted and returns scipy CSR transport plan."""
    rng = np.random.default_rng(99)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0, 1, (n, n))
    import scipy.sparse
    M_sp = scipy.sparse.csr_matrix(M_dense)
    G = sparse_ot.emd(a, b, M_sp, solver='lemon')
    assert scipy.sparse.issparse(G)
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=1)).ravel(), a, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=0)).ravel(), b, atol=1e-9
    )


def test_emd2_scipy_sparse_input():
    """emd2 with scipy CSR input returns same cost as POT."""
    rng = np.random.default_rng(55)
    n = 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M_dense = rng.uniform(0, 1, (n, n))
    import scipy.sparse
    M_sp = scipy.sparse.csr_matrix(M_dense)
    cost_sot = sparse_ot.emd2(a, b, M_sp, solver='lemon')
    cost_pot = ot.emd2(a, b, M_dense)
    assert abs(cost_sot - cost_pot) / abs(cost_pot) < 1e-6


def test_emd_ortools_raises_not_implemented():
    a, b, M = _problem(5, 5)
    with pytest.raises(NotImplementedError):
        sparse_ot.emd(a, b, M, solver='ortools')


def test_emd_invalid_solver_raises():
    a, b, M = _problem(5, 5)
    with pytest.raises(ValueError):
        sparse_ot.emd(a, b, M, solver='invalid')


def test_emd_cost_sparsity_threshold_drops_edges():
    """cost_sparsity_threshold drops low-cost edges; result has correct marginals."""
    rng = np.random.default_rng(11)
    n = 10
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    M = rng.uniform(0, 1, (n, n))
    G = sparse_ot.emd(a, b, M, cost_sparsity_threshold=0.2)
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/test_emd.py -v
```
Expected: all tests pass (12 original + 7 new = 19 passed).

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests across all test files pass.

- [ ] **Step 6: Commit**

```bash
git add src/sparse_ot/emd.py tests/test_emd.py
git commit -m "feat: emd() routes via routing.py; LEMON path + scipy sparse input support"
```

---

## Task 8: Final Verification

- [ ] **Step 1: Full test suite clean run**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass, 0 failures.

- [ ] **Step 2: Verify auto-routing selects expected solvers**

```bash
python - <<'EOF'
import numpy as np
from sparse_ot.routing import select_solver

# Small dense → LEMON (k=10 < 128)
print(select_solver(10, 10, 100))          # lemon

# Near-dense → Bonneel (k=200 > 128)
print(select_solver(200, 200, 200*200))    # bonneel

# Large sparse → ortools (n>1M)
print(select_solver(2_000_000, 100, 2_000_000 * 10))  # ortools
EOF
```
Expected output:
```
lemon
bonneel
ortools
```

- [ ] **Step 3: Commit final state**

```bash
git add -A
git status   # verify only expected files
git commit -m "feat: Plan 2 complete — LEMON CostScaling, routing, scipy sparse support"
```
