# Full-Basis Warm Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cold re-solve fallback in `refine_from_warm_start` with a full-basis warm start that injects prior dual potentials (and optionally the prior spanning tree) into the network simplex, giving speedup proportional to how close the prior solution is to optimal on `M_full`.

**Architecture:** Two C++ entry points (`solve_sparse_warm_potentials` for Mode C, `solve_sparse_warm_basis` for Mode B) added to `bonneel_solver.cpp`. Mode C calls `init()` then overrides `_pi`. Mode B skips `init()`'s artificial star and builds the spanning tree from `G_warm` instead, then calls `start()`. Python dispatch in `bonneel_sparse_solve_warm` (new helper in `sparse_utils.py`) selects the mode based on degeneracy, then `refine_from_warm_start` calls the helper instead of the cold `bonneel_sparse_solve`.

**Tech Stack:** C++14, pybind11, NumPy/SciPy (Python), pytest

---

## File Structure

| File | Change |
|---|---|
| `src/cpp/bonneel/network_simplex_simple.h` | Add `runWarmPotentials`, `warmBasisInit`, `runWarmBasis` methods |
| `src/cpp/bonneel_solver.cpp` | Add `solve_sparse_warm_potentials`, `solve_sparse_warm_basis` functions + pybind registration |
| `src/sparse_ot/sparse_utils.py` | Add `_DEGENERATE_WARN_THRESHOLD`, `bonneel_sparse_solve_warm` |
| `src/sparse_ot/refine.py` | Replace cold re-solve with `bonneel_sparse_solve_warm`; update `refine_info` |
| `tests/test_warm_basis.py` | New: C++ entry point unit tests |
| `tests/test_refine.py` | Add 6 new integration tests |
| `benchmarks/bench_refine.py` | Add `warm_basis_sec` / `warm_basis_amortized_sec` columns |

---

## Task 1: Mode C — `runWarmPotentials` in NetworkSimplexSimple

**Spec:** adds a public method that calls `init()` (artificial star), then overrides `_pi` from `(u0, v0)`, then calls `start()`. This is the fallback when `G_warm` is too degenerate for full-basis warm start.

**Files:**
- Modify: `src/cpp/bonneel/network_simplex_simple.h` — add method after `run()` at line ~814

**Sign convention (critical):** In `bonneel_solver.cpp`, duals are extracted as `up[i] = -net.potential(di(i))` and `vp[j] = net.potential(di(n+j))`. Internally `_pi` stores potentials for nodes. Source node `i` maps to `_pi[i]` via `_node_id`. Since `_node_id(n) = _node_num - n - 1` is only used for the graph API — and in `bonneel_solver.cpp`'s `solve_sparse` the extraction uses `di(i)` (source node index) directly — the mapping simplifies to: `_pi[i] = -u0[i]` for source nodes and `_pi[n+j] = v0[j]` for target nodes.

- [ ] **Step 1: Add `runWarmPotentials` to `network_simplex_simple.h`**

Insert the following after `run()` at line ~814 (inside the `public:` section):

```cpp
/// \brief Run with warm dual potentials (Mode C warm start).
///
/// Identical to run() but overrides the node potentials with the
/// provided warm values after init() builds the artificial spanning tree.
/// The pivot loop then starts from a better dual position.
///
/// \param u0  Source potentials, length n (0-indexed source nodes).
/// \param v0  Target potentials, length m (0-indexed target columns).
/// \param n   Number of source nodes.
/// \param m   Number of target nodes.
ProblemType runWarmPotentials(const double* u0, const double* v0, int n, int m) {
    if (!init()) return INFEASIBLE;
    // Override _pi with warm values. Root stays at 0.
    // Sign convention matches extraction in bonneel_solver.cpp:
    //   up[i] = -net.potential(di(i))  =>  _pi[i] = -u0[i]
    //   vp[j] =  net.potential(di(n+j)) =>  _pi[n+j] = v0[j]
    for (int i = 0; i < n; i++) _pi[i]     = -u0[i];
    for (int j = 0; j < m; j++) _pi[n + j] =  v0[j];
    return start();
}
```

- [ ] **Step 2: Add `solve_sparse_warm_potentials` to `bonneel_solver.cpp`**

Insert after `solve_sparse` (before `PYBIND11_MODULE`):

```cpp
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse_warm_potentials(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    py::array_t<double, py::array::c_style | py::array::forcecast> u0,
    py::array_t<double, py::array::c_style | py::array::forcecast> v0,
    int numItermax
) {
    auto a_buf  = a.request();  auto b_buf  = b.request();
    auto rp_buf = row_ptr.request(); auto ci_buf = col_idx.request();
    auto c_buf  = costs.request();
    auto u0_buf = u0.request(); auto v0_buf = v0.request();

    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const int64_t k = static_cast<int64_t>(c_buf.size);
    const double* ap  = static_cast<const double*>(a_buf.ptr);
    const double* bp  = static_cast<const double*>(b_buf.ptr);
    const int*    rp  = static_cast<const int*>(rp_buf.ptr);
    const int*    ci  = static_cast<const int*>(ci_buf.ptr);
    const double* cp  = static_cast<const double*>(c_buf.ptr);
    const double* u0p = static_cast<const double*>(u0_buf.ptr);
    const double* v0p = static_cast<const double*>(v0_buf.ptr);

    py::array_t<double> u(n), v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int> rows(0), cols(0); py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0); std::fill(vp, vp + m, 0.0);
        return std::make_tuple(rows, cols, vals, u, v);
    }

    BipartiteSparseDigraph di(n, m, rp, ci, k);
    NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(
        di, true, n + m, k, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int64_t i = 0; i < k; i++)
        net.setCost(BipartiteSparseDigraph::arcFromId(i), cp[i]);

    net.runWarmPotentials(u0p, v0p, n, m);

    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k); out_cols.reserve(k); out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }
    py::array_t<int>    rows_out(out_rows.size()), cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(), out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(), out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(), out_vals.size() * sizeof(double));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}
```

- [ ] **Step 3: Register `solve_sparse_warm_potentials` in `PYBIND11_MODULE`**

Inside the `PYBIND11_MODULE(_bonneel, m)` block, add after the `solve_sparse` registration:

```cpp
m.def("solve_sparse_warm_potentials", &solve_sparse_warm_potentials,
      py::arg("a"), py::arg("b"),
      py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
      py::arg("u0"), py::arg("v0"),
      py::arg("numItermax") = 100000,
      "Solve sparse OT with warm dual potentials (Mode C). "
      "Returns (rows, cols, vals, u, v).");
```

- [ ] **Step 4: Rebuild the C++ extension**

```bash
cd /Users/jonatanbobrutsky-haim/Documents/Code/sparse-optimal-transport
pip install -e . --no-build-isolation -q
```

Expected: no errors, `_bonneel` module rebuilt.

- [ ] **Step 5: Smoke-test the new binding**

```python
python -c "
from sparse_ot._ext import _bonneel
print(dir(_bonneel))
assert hasattr(_bonneel, 'solve_sparse_warm_potentials'), 'missing'
print('OK')
"
```

Expected: `solve_sparse_warm_potentials` in the printed list, `OK` printed.

- [ ] **Step 6: Commit**

```bash
git add src/cpp/bonneel/network_simplex_simple.h src/cpp/bonneel_solver.cpp
git commit -m "feat(warm-basis): Mode C — runWarmPotentials + solve_sparse_warm_potentials"
```

---

## Task 2: Python dispatch helper (Mode C only) + `refine.py` update

**Spec:** `bonneel_sparse_solve_warm` in `sparse_utils.py` dispatches to Mode C only (Mode B added in Task 6). `refine_from_warm_start` replaces the cold re-solve with this helper. `refine_info` gains `warm_basis_used`.

**Files:**
- Modify: `src/sparse_ot/sparse_utils.py`
- Modify: `src/sparse_ot/refine.py`

- [ ] **Step 1: Write a failing test for the warm dispatch in `tests/test_warm_basis.py` (create file)**

```python
"""Tests for warm-started network simplex entry points."""
import numpy as np
import pytest
import scipy.sparse

from sparse_ot import emd
from sparse_ot.sparse_utils import bonneel_sparse_solve_warm, to_csr


def _band_csr(n, k, seed=0):
    rng = np.random.default_rng(seed)
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + k); lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i); cols.append(j)
            costs.append(float((i - j) ** 2) + rng.uniform(0, 0.1))
    M = scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )
    a = np.ones(n) / n
    b = np.ones(n) / n
    return a, b, M


def test_warm_potentials_mode_c_correct():
    """Mode C produces the same optimal cost as a cold solve."""
    n = 40
    a, b, M = _band_csr(n, k=8)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    G_warm, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, nn, mm,
        G_cold, info["u"], info["v"],
    )
    warm_cost = float(G_warm.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is False  # Mode B not yet implemented
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_warm_basis.py::test_warm_potentials_mode_c_correct -v
```

Expected: `ImportError` or `AttributeError` — `bonneel_sparse_solve_warm` not yet defined.

- [ ] **Step 3: Add `bonneel_sparse_solve_warm` to `sparse_utils.py`**

Add after the existing `bonneel_sparse_solve` function (and add `import warnings` at the top if not present):

```python
_DEGENERATE_WARN_THRESHOLD = 0.05  # fraction of n+m-1


def bonneel_sparse_solve_warm(a, b, row_ptr, col_idx, costs, n, m,
                               G_warm, u0, v0, numItermax=None):
    """Warm-started network simplex: full basis (Mode B) or potential-only (Mode C).

    Mode B (full basis) is used when G_warm has enough non-degenerate arcs to
    reconstruct a spanning tree. Mode C (potential-only) is the fallback.

    Returns
    -------
    G    : CSR scipy matrix (n, m)
    u    : float64 1-D, shape (n,)
    v    : float64 1-D, shape (m,)
    warm_basis_used : bool — True if Mode B fired, False if Mode C fired.
    """
    from sparse_ot._ext import _bonneel

    if numItermax is None:
        numItermax = _default_num_iter(n, m, len(costs))
    else:
        numItermax = int(numItermax)

    n_basic = n + m - 1

    # 1. Drop stored zeros (inflated nnz from scipy CSR bookkeeping).
    G_warm = G_warm.copy()
    G_warm.eliminate_zeros()

    n_degenerate = n_basic - G_warm.nnz

    if n_degenerate < 0:
        # G_warm is non-basic (user-constructed, perturbed, etc.).
        warnings.warn(
            f"warm_start G has {-n_degenerate} more nonzeros than a spanning tree "
            f"(nnz={G_warm.nnz}, n+m-1={n_basic}). G_warm is not a basic feasible "
            "solution; arcs sorted by flow and spanning tree extracted.",
            RuntimeWarning, stacklevel=4,
        )
        n_degenerate = 0  # treat as non-degenerate; flow sort handles quality

    # 2. Mode selection.
    if n_degenerate > _DEGENERATE_WARN_THRESHOLD * n_basic:
        warnings.warn(
            f"warm_start G has {n_degenerate} zero-flow basic arcs "
            f"({100 * n_degenerate / n_basic:.0f}% of the spanning tree). "
            "Basis reconstruction will be approximate; using potential-only warm start.",
            RuntimeWarning, stacklevel=4,
        )
        warm_basis_used = False
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_potentials(
            a, b, row_ptr, col_idx, costs,
            np.asarray(u0, dtype=np.float64),
            np.asarray(v0, dtype=np.float64),
            numItermax,
        )
    else:
        # Mode B — full basis. Sort arcs by flow descending so union-find
        # in C++ prefers high-flow (likely basic) arcs as tree arcs.
        coo = G_warm.tocoo()
        order = np.argsort(coo.data)[::-1]
        warm_rows = coo.row[order].astype(np.int32)
        warm_cols = coo.col[order].astype(np.int32)
        warm_flows = coo.data[order].astype(np.float64)

        # Compute CSR arc IDs for each warm arc via flat-key searchsorted.
        # row_ptr / col_idx are already sorted within each row (scipy guarantee).
        n_rows = n
        n_cols = m
        src_per_arc = np.repeat(
            np.arange(n_rows, dtype=np.int64), np.diff(row_ptr.astype(np.int64))
        )
        M_keys = src_per_arc * n_cols + col_idx.astype(np.int64)
        warm_keys = warm_rows.astype(np.int64) * n_cols + warm_cols.astype(np.int64)
        arc_ids = np.searchsorted(M_keys, warm_keys).astype(np.int32)

        warm_basis_used = True
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_basis(
            a, b, row_ptr, col_idx, costs,
            np.asarray(u0, dtype=np.float64),
            np.asarray(v0, dtype=np.float64),
            arc_ids, warm_rows, warm_cols, warm_flows,
            numItermax,
        )

    G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    return G, u, v, warm_basis_used
```

Also add `import warnings` at the top of `sparse_utils.py` if not already present.

- [ ] **Step 4: Run the test (expect pass with `basis_used is False`)**

```bash
pytest tests/test_warm_basis.py::test_warm_potentials_mode_c_correct -v
```

Expected: PASS. The Mode B branch raises `AttributeError` on `solve_sparse_warm_basis` but the Mode C branch runs (G_warm from a prior solve is non-degenerate but we force `basis_used=False` via the assertion).

Wait — G_warm from a cold solve has `nnz ≤ n+m-1`, so `n_degenerate ≥ 0`. If `n_degenerate = 0` (non-degenerate), the else branch fires and tries `solve_sparse_warm_basis` which doesn't exist yet. Fix the test to force Mode C:

Update the test:
```python
def test_warm_potentials_mode_c_correct():
    """Mode C produces the same optimal cost as a cold solve."""
    import warnings
    n = 40
    a, b, M = _band_csr(n, k=8)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    # Construct a severely degenerate G_warm to force Mode C path.
    # Keep only 1 arc — n_degenerate >> 5% threshold.
    coo = G_cold.tocoo()
    idx = np.argmax(coo.data)
    G_degenerate = scipy.sparse.csr_matrix(
        ([coo.data[idx]], ([coo.row[idx]], [coo.col[idx]])), shape=M.shape
    )

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        G_warm, u, v, basis_used = bonneel_sparse_solve_warm(
            a, b, row_ptr, col_idx, costs, nn, mm,
            G_degenerate, info["u"], info["v"],
        )
    warm_cost = float(G_warm.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is False
```

- [ ] **Step 5: Run updated test**

```bash
pytest tests/test_warm_basis.py::test_warm_potentials_mode_c_correct -v
```

Expected: PASS.

- [ ] **Step 6: Update `refine.py` — non-optimal branch**

In `refine_from_warm_start`, replace:

```python
        G, u, v = bonneel_sparse_solve(a, b, row_ptr, col_idx, costs, n, m, numItermax)
        edges_added = int(G.nnz - G_warm.nnz)
        refine_info = {
            "warm_start_optimal": False,
            "num_passes": 1,
            "initial_min_reduced_cost": min_rc,
            "edges_added": max(edges_added, 0),
        }
```

with:

```python
        from sparse_ot.sparse_utils import bonneel_sparse_solve_warm
        G, u, v, warm_basis_used = bonneel_sparse_solve_warm(
            a, b, row_ptr, col_idx, costs, n, m, G_warm, u, v, numItermax
        )
        edges_added = int(G.nnz - G_warm.nnz)
        refine_info = {
            "warm_start_optimal": False,
            "num_passes": 1,
            "initial_min_reduced_cost": min_rc,
            "edges_added": max(edges_added, 0),
            "warm_basis_used": warm_basis_used,
        }
```

Also update the import at the top of `refine.py` — remove `bonneel_sparse_solve` from the imports (it's no longer called directly from refine.py):

```python
from sparse_ot.sparse_utils import to_csr, _MARGINAL_TOL
```

- [ ] **Step 7: Run full test suite to verify nothing broke**

```bash
pytest tests/ -x -q
```

Expected: all existing tests pass. The non-optimal branch now goes through `bonneel_sparse_solve_warm` → Mode C (for degenerate G_warm in tests) or Mode B (will fail until Task 5 — but existing tests don't use the non-optimal branch with non-degenerate G_warm directly in a way that hits the new code path without proper setup). Verify all 79+ tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/sparse_ot/sparse_utils.py src/sparse_ot/refine.py tests/test_warm_basis.py
git commit -m "feat(warm-basis): Python dispatch helper + refine.py non-optimal branch update"
```

---

## Task 3: Mode B C++ core — `warmBasisInit` + `runWarmBasis`

**Spec:** `warmBasisInit` replaces `init()`'s artificial-star block. It sets `_pi`, runs union-find on the warm arcs to build a spanning forest, constructs DFS thread arrays, attaches any unspanned nodes to root via artificial arcs, then sets arc states. `runWarmBasis` = `warmBasisInit` + `start()`.

**Files:**
- Modify: `src/cpp/bonneel/network_simplex_simple.h`

The method signature uses pre-computed arc IDs (CSR positions) from Python, so the C++ doesn't need `col_idx` lookups internally.

- [ ] **Step 1: Read `init()` in `network_simplex_simple.h` lines 1058–1210 carefully**

Understand: `_search_arc_num`, `_all_arc_num`, ART_COST computation, `_source`/`_target`/`_cost`/`_flow`/`_state` for artificial arcs. `warmBasisInit` must replicate the non-tree portions of `init()` and set `_search_arc_num` / `_all_arc_num` correctly.

- [ ] **Step 2: Add `warmBasisInit` and `runWarmBasis` to `network_simplex_simple.h`**

Insert inside the `public:` section after `runWarmPotentials`:

```cpp
/// \brief Warm-start with full basis injection (Mode B).
///
/// Replaces init()'s artificial-star block. Union-find builds a spanning
/// tree from the warm arcs; unspanned nodes are attached via artificial arcs.
/// \param u0         Source potentials (length n).
/// \param v0         Target potentials (length m).
/// \param arc_ids    CSR arc IDs for each warm arc (length n_warm).
/// \param warm_src   Source node index 0..n-1 for each warm arc.
/// \param warm_tgt   Target column index 0..m-1 for each warm arc (NOT n+col).
/// \param warm_flow  Flow value for each warm arc.
/// \param n_warm     Number of warm arcs (G_warm.nnz after eliminate_zeros).
/// \param n          Number of source nodes.
/// \param m          Number of target nodes.
bool warmBasisInit(
    const double* u0, const double* v0,
    const int* arc_ids, const int* warm_src, const int* warm_tgt,
    const double* warm_flow, int n_warm, int n, int m
) {
    if (_node_num == 0) return false;

    // --- Replicate non-tree portions of init() ---
    _sum_supply = 0;
    for (int i = 0; i != _node_num; ++i) _sum_supply += _supply[i];

    // ART_COST (same formula as init())
    Cost ART_COST;
    if (std::numeric_limits<Cost>::is_exact) {
        ART_COST = std::numeric_limits<Cost>::max() / 2 + 1;
    } else {
        ART_COST = 0;
        for (ArcsType i = 0; i != _arc_num; ++i)
            if (_cost[i] > ART_COST) ART_COST = _cost[i];
        ART_COST = (ART_COST + 1) * _node_num;
    }

    // Initialize real arc states to LOWER; clear flow.
    for (ArcsType i = 0; i != _arc_num; ++i)
        _state[i] = STATE_LOWER;
#ifdef SPARSE_FLOW
    _flow = SparseValueVector<Value>();
#else
    for (ArcsType i = 0; i != _arc_num; ++i) _flow[i] = 0;
#endif

    // Root node (same as init())
    _search_arc_num = _arc_num;
    _all_arc_num    = _arc_num + _node_num;  // EQ supply assumed (balanced OT)
    _root = _node_num;
    _parent[_root] = -1;
    _pred[_root]   = -1;
    _supply[_root] = -_sum_supply;
    _pi[_root]     = 0;

    // --- Set warm potentials ---
    for (int i = 0; i < n; i++) _pi[i]     = -u0[i];
    for (int j = 0; j < m; j++) _pi[n + j] =  v0[j];

    // --- Union-Find over real nodes 0.._node_num-1 ---
    int node_num = _node_num;
    std::vector<int> uf(node_num + 1);  // +1 for _root slot (unused)
    std::iota(uf.begin(), uf.end(), 0);
    auto uf_find = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };

    // per-node tree info (indexed 0..node_num-1; root handled separately)
    std::vector<int>      par(node_num, _root);
    std::vector<ArcsType> pred_arc(node_num, ArcsType(-1));
    std::vector<bool>     fwd(node_num, true);
    std::vector<std::vector<int>> children(node_num + 1);  // children[u] list

    // Process warm arcs (already sorted by flow desc from Python)
    for (int k = 0; k < n_warm; k++) {
        int s = warm_src[k];
        int t = n + warm_tgt[k];   // target node in [n, n+m)
        int rs = uf_find(s), rt = uf_find(t);
        if (rs == rt) continue;    // would form a cycle — skip

        // Merge components: orient s as child of t
        uf[rs] = rt;
        par[s]      = t;
        pred_arc[s] = arc_ids[k];
        fwd[s]      = true;        // arc goes s→t (source to target)
        children[t].push_back(s);
        _state[arc_ids[k]] = STATE_TREE;
        _flow[arc_ids[k]]  = static_cast<Value>(warm_flow[k]);
    }

    // Attach unspanned real nodes to _root via artificial arcs
    for (int u = 0; u < node_num; u++) {
        if (uf_find(u) != uf_find(_root)) {
            uf[uf_find(u)] = _root;  // merge with root component
            children[_root].push_back(u);
            ArcsType e = _arc_num + u;
            par[u]      = _root;
            pred_arc[u] = e;
            // Direction matches init(): supply>=0 → arc u→root (forward),
            //                           supply<0  → arc root→u (backward)
            if (_supply[u] >= 0) {
                fwd[u]      = true;
                _source[e]  = u; _target[e] = _root;
                _flow[e]    = static_cast<Value>(_supply[u]);
                _cost[e]    = 0;
            } else {
                fwd[u]      = false;
                _source[e]  = _root; _target[e] = u;
                _flow[e]    = static_cast<Value>(-_supply[u]);
                _cost[e]    = ART_COST;
            }
            _state[e] = STATE_TREE;
        }
    }

    // Also attach _root's children that came via warm arcs but whose
    // parent was set to a non-root node — verify _root is reachable.
    // (Already guaranteed: every component was merged into uf_find(_root).)

    // --- Build DFS thread list from _root ---
    // Pre-order DFS: _thread[u] = next node in DFS traversal order.
    // Post-order: _succ_num[u], _last_succ[u].
    // We do two passes: (1) pre-order for _thread/_rev_thread,
    //                   (2) post-order for _succ_num/_last_succ.

    // Pass 1: iterative pre-order DFS
    {
        int prev_node = _root;
        std::stack<int> stk;
        // Push _root's children in reverse so left-most child is processed first
        for (int i = (int)children[_root].size() - 1; i >= 0; --i)
            stk.push(children[_root][i]);
        while (!stk.empty()) {
            int u = stk.top(); stk.pop();
            _thread[prev_node]  = u;
            _rev_thread[u]      = prev_node;
            prev_node = u;
            for (int i = (int)children[u].size() - 1; i >= 0; --i)
                stk.push(children[u][i]);
        }
        // Last node's thread goes back to _root (circular thread)
        _thread[prev_node] = _root;
        _rev_thread[_root] = prev_node;
        // _root's own thread successor was already set to first child above.
        // If _root has no children, thread[_root] = _root.
        if (children[_root].empty()) {
            _thread[_root]    = _root;
            _rev_thread[_root] = _root;
        }
    }

    // Pass 2: iterative post-order DFS for _succ_num and _last_succ
    {
        // Use explicit stack with "visited" flag for post-order
        std::stack<std::pair<int,bool>> stk;
        stk.push({_root, false});
        while (!stk.empty()) {
            auto [u, processed] = stk.top(); stk.pop();
            if (processed) {
                if (children[u].empty()) {
                    _succ_num[u]  = 1;
                    _last_succ[u] = u;
                } else {
                    _succ_num[u]  = 1;
                    _last_succ[u] = _last_succ[children[u].back()];
                    for (int c : children[u]) _succ_num[u] += _succ_num[c];
                }
            } else {
                stk.push({u, true});
                for (int i = (int)children[u].size() - 1; i >= 0; --i)
                    stk.push({children[u][i], false});
            }
        }
    }

    // Set _parent, _pred, _forward for all real nodes
    for (int u = 0; u < node_num; u++) {
        _parent[u]  = par[u];
        _pred[u]    = pred_arc[u];
        _forward[u] = fwd[u];
    }

    return true;
}

/// \brief Run with full-basis warm start (Mode B).
ProblemType runWarmBasis(
    const double* u0, const double* v0,
    const int* arc_ids, const int* warm_src, const int* warm_tgt,
    const double* warm_flow, int n_warm, int n, int m
) {
    if (!warmBasisInit(u0, v0, arc_ids, warm_src, warm_tgt, warm_flow, n_warm, n, m))
        return INFEASIBLE;
    return start();
}
```

**Note:** This implementation assumes balanced OT (`_sum_supply == 0`, i.e. EQ supply type), which is the only case the solver currently handles. The `_all_arc_num = _arc_num + _node_num` line mirrors the EQ branch of `init()`.

- [ ] **Step 3: Rebuild**

```bash
pip install -e . --no-build-isolation -q
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/cpp/bonneel/network_simplex_simple.h
git commit -m "feat(warm-basis): warmBasisInit + runWarmBasis in NetworkSimplexSimple"
```

---

## Task 4: `solve_sparse_warm_basis` C++ entry point + pybind registration

**Spec:** mirrors `solve_sparse_warm_potentials` but calls `runWarmBasis` instead of `runWarmPotentials`. Receives pre-computed arc IDs from Python.

**Files:**
- Modify: `src/cpp/bonneel_solver.cpp`

- [ ] **Step 1: Add `solve_sparse_warm_basis` to `bonneel_solver.cpp`**

Insert after `solve_sparse_warm_potentials`:

```cpp
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse_warm_basis(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    py::array_t<double, py::array::c_style | py::array::forcecast> u0,
    py::array_t<double, py::array::c_style | py::array::forcecast> v0,
    py::array_t<int,    py::array::c_style | py::array::forcecast> arc_ids,
    py::array_t<int,    py::array::c_style | py::array::forcecast> warm_src,
    py::array_t<int,    py::array::c_style | py::array::forcecast> warm_tgt,
    py::array_t<double, py::array::c_style | py::array::forcecast> warm_flow,
    int numItermax
) {
    auto a_buf   = a.request();   auto b_buf   = b.request();
    auto rp_buf  = row_ptr.request(); auto ci_buf = col_idx.request();
    auto c_buf   = costs.request();
    auto u0_buf  = u0.request();  auto v0_buf  = v0.request();
    auto aid_buf = arc_ids.request();
    auto ws_buf  = warm_src.request(); auto wt_buf = warm_tgt.request();
    auto wf_buf  = warm_flow.request();

    const int n    = static_cast<int>(a_buf.size);
    const int m    = static_cast<int>(b_buf.size);
    const int64_t k = static_cast<int64_t>(c_buf.size);
    const int n_warm = static_cast<int>(aid_buf.size);

    const double* ap   = static_cast<const double*>(a_buf.ptr);
    const double* bp   = static_cast<const double*>(b_buf.ptr);
    const int*    rp   = static_cast<const int*>(rp_buf.ptr);
    const int*    ci   = static_cast<const int*>(ci_buf.ptr);
    const double* cp   = static_cast<const double*>(c_buf.ptr);
    const double* u0p  = static_cast<const double*>(u0_buf.ptr);
    const double* v0p  = static_cast<const double*>(v0_buf.ptr);
    const int*    aidp = static_cast<const int*>(aid_buf.ptr);
    const int*    wsp  = static_cast<const int*>(ws_buf.ptr);
    const int*    wtp  = static_cast<const int*>(wt_buf.ptr);
    const double* wfp  = static_cast<const double*>(wf_buf.ptr);

    py::array_t<double> u(n), v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int> rows(0), cols(0); py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0); std::fill(vp, vp + m, 0.0);
        return std::make_tuple(rows, cols, vals, u, v);
    }

    BipartiteSparseDigraph di(n, m, rp, ci, k);
    NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(
        di, true, n + m, k, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int64_t i = 0; i < k; i++)
        net.setCost(BipartiteSparseDigraph::arcFromId(i), cp[i]);

    net.runWarmBasis(u0p, v0p, aidp, wsp, wtp, wfp, n_warm, n, m);

    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k); out_cols.reserve(k); out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }
    py::array_t<int>    rows_out(out_rows.size()), cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(), out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(), out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(), out_vals.size() * sizeof(double));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}
```

- [ ] **Step 2: Register in `PYBIND11_MODULE`**

```cpp
m.def("solve_sparse_warm_basis", &solve_sparse_warm_basis,
      py::arg("a"), py::arg("b"),
      py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
      py::arg("u0"), py::arg("v0"),
      py::arg("arc_ids"), py::arg("warm_src"), py::arg("warm_tgt"),
      py::arg("warm_flow"),
      py::arg("numItermax") = 100000,
      "Solve sparse OT with full-basis warm start (Mode B). "
      "Returns (rows, cols, vals, u, v).");
```

- [ ] **Step 3: Rebuild**

```bash
pip install -e . --no-build-isolation -q
```

- [ ] **Step 4: Smoke-test**

```python
python -c "
from sparse_ot._ext import _bonneel
assert hasattr(_bonneel, 'solve_sparse_warm_basis'), 'missing'
print('OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/cpp/bonneel_solver.cpp
git commit -m "feat(warm-basis): solve_sparse_warm_basis C++ entry point + pybind"
```

---

## Task 5: Mode B correctness tests

**Spec:** verify Mode B produces the correct optimum in three scenarios: same support / same metric, same support / perturbed metric, and subset support (the canonical refine case).

**Files:**
- Modify: `tests/test_warm_basis.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_warm_basis.py`:

```python
def test_warm_basis_mode_b_same_support_same_metric():
    """Mode B: same support + same metric → cost matches cold, basis_used=True."""
    n = 50
    a, b, M = _band_csr(n, k=10, seed=1)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, nn, mm,
        G_cold, info["u"], info["v"],
    )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is True


def test_warm_basis_mode_b_same_support_perturbed_metric():
    """Mode B: same support, perturbed costs → result matches cold on perturbed M."""
    import copy
    n = 50
    a, b, M1 = _band_csr(n, k=10, seed=2)
    G1, info1 = emd(a, b, M1, log=True)

    # Perturb costs slightly (same sparsity pattern)
    rng = np.random.default_rng(42)
    M2_data = M1.data + rng.uniform(-0.01, 0.01, size=M1.nnz)
    M2 = scipy.sparse.csr_matrix((M2_data, M1.indices.copy(), M1.indptr.copy()), shape=M1.shape)
    G2_cold, info2_cold = emd(a, b, M2, log=True)
    cold_cost2 = info2_cold["cost"]

    row_ptr, col_idx, costs2, nn, mm, _ = to_csr(M2, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs2, nn, mm,
        G1, info1["u"], info1["v"],
    )
    warm_cost = float(G_w.multiply(M2).sum())
    assert abs(warm_cost - cold_cost2) < 1e-9
    assert basis_used is True


def test_warm_basis_mode_b_subset_support():
    """Mode B: warm support ⊂ M_full support → matches cold solve on M_full."""
    n = 60
    a, b, M_full = _band_csr(n, k=12, seed=3)

    # Phase 1: solve on tighter k=6 band
    half = 3
    rows6, cols6, costs6 = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + 6); lo = max(0, hi - 6)
        for j in range(lo, hi):
            rows6.append(i); cols6.append(j); costs6.append(float((i-j)**2))
    M6 = scipy.sparse.csr_matrix((costs6, (rows6, cols6)), shape=(n, n))
    G6, info6 = emd(a, b, M6, log=True)

    G_cold, info_cold = emd(a, b, M_full, log=True)
    cold_cost = info_cold["cost"]

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M_full, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, nn, mm,
        G6, info6["u"], info6["v"],
    )
    warm_cost = float(G_w.multiply(M_full).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    assert basis_used is True


def test_warm_basis_sign_convention():
    """2x2 analytic problem: warm-start duals match the known optimal duals."""
    # 2x2 balanced OT: a=[0.5,0.5], b=[0.5,0.5]
    # M = [[1, 2], [3, 1]] — optimal plan: diagonal, cost = 1.0
    # Optimal duals (one solution): u=[0,0], v=[1,1]
    M = scipy.sparse.csr_matrix(np.array([[1., 2.], [3., 1.]]))
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    G_cold, info = emd(a, b, M, log=True, center_dual=False)
    cold_cost = info["cost"]
    assert abs(cold_cost - 1.0) < 1e-9

    row_ptr, col_idx, costs, n, m, _ = to_csr(M, 0.0)
    G_w, u, v, basis_used = bonneel_sparse_solve_warm(
        a, b, row_ptr, col_idx, costs, n, m,
        G_cold, info["u"], info["v"],
    )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9


def test_warm_basis_non_basic_G_warns_and_correct():
    """Non-basic G_warm (nnz > n+m-1): warning emitted, result still correct."""
    import warnings
    n = 30
    a, b, M = _band_csr(n, k=8, seed=5)
    G_cold, info = emd(a, b, M, log=True)
    cold_cost = info["cost"]

    # Make G_warm non-basic by combining two plans
    rng = np.random.default_rng(7)
    a2 = np.abs(rng.standard_normal(n)); a2 /= a2.sum()
    b2 = np.abs(rng.standard_normal(n)); b2 /= b2.sum()
    # Use a different b to get a different plan on the same M
    # (just manually inflate G_cold with a few extra off-diagonal entries)
    coo = G_cold.tocoo()
    # Add a tiny entry somewhere else in M's support to inflate nnz
    extra_r = coo.row[0]; extra_c = (coo.col[0] + 1) % n
    if M[extra_r, extra_c] > 0:  # only if that edge is in M's support
        G_nonbasic = scipy.sparse.csr_matrix(
            (np.append(coo.data, 1e-15),
             (np.append(coo.row, extra_r), np.append(coo.col, extra_c))),
            shape=M.shape
        )
    else:
        G_nonbasic = G_cold  # skip test if edge not in support

    row_ptr, col_idx, costs, nn, mm, _ = to_csr(M, 0.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G_w, u, v, basis_used = bonneel_sparse_solve_warm(
            a, b, row_ptr, col_idx, costs, nn, mm,
            G_nonbasic, info["u"], info["v"],
        )
    warm_cost = float(G_w.multiply(M).sum())
    assert abs(warm_cost - cold_cost) < 1e-9
    # Check warning was emitted if G_nonbasic has extra entries
    if G_nonbasic.nnz > n + n - 1:
        assert any("more nonzeros" in str(warning.message) for warning in w)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_warm_basis.py -v
```

Expected: all tests PASS. If any fail, check sign convention and `_pi` indexing in `warmBasisInit`.

- [ ] **Step 3: Run full suite**

```bash
pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_warm_basis.py
git commit -m "test(warm-basis): Mode B correctness + sign convention + non-basic G_warm"
```

---

## Task 6: Integration tests via `emd(warm_start=...)` + refine_info check

**Spec:** verify the end-to-end path through `emd()` → `refine_from_warm_start` → `bonneel_sparse_solve_warm` produces correct `refine_info["warm_basis_used"]` and costs.

**Files:**
- Modify: `tests/test_refine.py`

- [ ] **Step 1: Add tests to `tests/test_refine.py`**

```python
def test_refine_non_optimal_warm_basis_used():
    """Non-optimal branch: warm_basis_used=True when G_warm is non-degenerate."""
    import scipy.sparse
    n = 40
    rng = np.random.default_rng(99)
    half = 3
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i-half); hi = min(n, lo+6); lo = max(0, hi-6)
        for j in range(lo, hi):
            rows.append(i); cols.append(j); costs.append(float((i-j)**2)+0.01)
    M6 = scipy.sparse.csr_matrix((costs, (rows, cols)), shape=(n, n))

    half2 = 5
    rows2, cols2, costs2 = [], [], []
    for i in range(n):
        lo = max(0, i-half2); hi = min(n, lo+10); lo = max(0, hi-10)
        for j in range(lo, hi):
            rows2.append(i); cols2.append(j); costs2.append(float((i-j)**2)+0.01)
    M10 = scipy.sparse.csr_matrix((costs2, (rows2, cols2)), shape=(n, n))

    a = np.ones(n) / n
    b = np.ones(n) / n

    from sparse_ot import emd
    G6, info6 = emd(a, b, M6, log=True)
    G10_cold, info10_cold = emd(a, b, M10, log=True)
    G10_warm, info10_warm = emd(a, b, M10, warm_start=(G6, info6), log=True)

    assert abs(info10_warm["cost"] - info10_cold["cost"]) < 1e-9
    refine = info10_warm["refine"]
    assert refine["warm_start_optimal"] is False
    assert refine["warm_basis_used"] is True  # non-degenerate G6


def test_refine_non_optimal_mode_c_fallback():
    """Non-optimal branch: warm_basis_used=False when G_warm is degenerate."""
    import warnings, scipy.sparse
    n = 30
    rows, cols, costs = [], [], []
    for i in range(n):
        for j in range(max(0,i-2), min(n,i+3)):
            rows.append(i); cols.append(j); costs.append(float((i-j)**2)+0.1)
    M = scipy.sparse.csr_matrix((costs, (rows, cols)), shape=(n, n))
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G_warm, info_warm = emd(a, b, M, warm_start=(G_deg, info), log=True)
    assert abs(info_warm["cost"] - cold_cost) < 1e-9
    refine = info_warm["refine"]
    assert refine["warm_basis_used"] is False
    assert any("zero-flow basic arcs" in str(x.message) for x in w)
```

- [ ] **Step 2: Run new tests**

```bash
pytest tests/test_refine.py::test_refine_non_optimal_warm_basis_used \
       tests/test_refine.py::test_refine_non_optimal_mode_c_fallback -v
```

Expected: both PASS.

- [ ] **Step 3: Run full suite**

```bash
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_refine.py
git commit -m "test(warm-basis): integration tests for warm_basis_used flag via emd()"
```

---

## Task 7: Benchmark additions

**Spec:** add `warm_basis_sec` and `warm_basis_amortized_sec` columns to each benchmark cell in `bench_refine.py`. The existing correctness gate covers the new path.

**Files:**
- Modify: `benchmarks/bench_refine.py`

- [ ] **Step 1: Update `_run_cell` in `benchmarks/bench_refine.py`**

Replace the existing `_run_cell` function with:

```python
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

    # Warm-basis timing: phase 1 same as above, only re-time phase 2
    # (same call as refine — warm_basis_sec just aliases refine path for clarity)
    warm_basis_sec = t_phase1 + t_phase2
    warm_basis_amortized_sec = t_phase2

    # Correctness gate
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
        "warm_basis_sec": warm_basis_sec,
        "warm_basis_amortized_sec": warm_basis_amortized_sec,
        "warm_start_optimal": info_refined["refine"]["warm_start_optimal"],
        "warm_basis_used": info_refined["refine"].get("warm_basis_used", None),
        "edges_added": info_refined["refine"]["edges_added"],
    }
```

- [ ] **Step 2: Run quick benchmark to verify it works**

```bash
python benchmarks/bench_refine.py --quick
```

Expected: writes `benchmarks/results/refine_quick.json` with `warm_basis_used` field in each cell. No errors.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_refine.py benchmarks/results/refine_quick.json
git commit -m "bench(warm-basis): add warm_basis_sec and warm_basis_used columns"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Mode C: `runWarmPotentials` in NSS | Task 1 |
| Mode C: `solve_sparse_warm_potentials` pybind | Task 1 |
| Mode B: `warmBasisInit` + `runWarmBasis` in NSS | Task 3 |
| Mode B: `solve_sparse_warm_basis` pybind | Task 4 |
| `bonneel_sparse_solve_warm` Python dispatch | Task 2 |
| `eliminate_zeros` + non-basic check | Task 2 |
| Descending-flow sort before Mode B | Task 2 |
| Arc ID computation via searchsorted | Task 2 |
| `_DEGENERATE_WARN_THRESHOLD` constant | Task 2 |
| `refine_from_warm_start` non-optimal branch update | Task 2 |
| `refine_info["warm_basis_used"]` | Task 2 |
| Mode C correctness test | Task 2 |
| Mode B correctness tests (3 scenarios) | Task 5 |
| Sign convention test | Task 5 |
| Non-basic G_warm warn + correct | Task 5 |
| Integration via `emd()` + refine_info | Task 6 |
| Benchmark columns | Task 7 |

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:**
- `bonneel_sparse_solve_warm` returns `(G, u, v, warm_basis_used: bool)` — consistent across Task 2 (implementation) and Tasks 5-6 (tests).
- `warmBasisInit` signature `(u0, v0, arc_ids, warm_src, warm_tgt, warm_flow, n_warm, n, m)` — consistent between Task 3 (C++ implementation) and Task 4 (`solve_sparse_warm_basis` call).
- `refine_info["warm_basis_used"]` — added in Task 2, tested in Task 6.
- Build step precedes all test runs in every task.
