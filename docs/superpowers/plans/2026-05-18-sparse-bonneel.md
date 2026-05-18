# Sparse Bonneel + Solver Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sparse-input path to Bonneel's `NetworkSimplexSimple` (O(k+n+m) memory), expose dual potentials via a POT-compatible `log=True` dict, and consolidate the solver surface to {Bonneel-dense, Bonneel-sparse} by deleting LEMON CostScaling and OR-Tools.

**Architecture:** Re-instantiate the existing `NetworkSimplexSimple` template on a new minimal sparse bipartite digraph (CSR-backed view); no fork of the simplex code. Public Python API matches POT's `ot.emd` / `ot.emd2` exactly.

**Tech Stack:** C++17, pybind11, NumPy, SciPy, pytest. Existing Bonneel vendored headers (`src/cpp/bonneel/network_simplex_simple.h`, `full_bipartitegraph.h`).

**Spec:** `docs/superpowers/specs/2026-05-18-sparse-bonneel-design.md`

---

## File Structure

**New files:**
- `src/cpp/bonneel/bipartite_sparse_digraph.h` — CSR-backed LEMON-Digraph-concept view (~120 LoC).
- `tests/test_bonneel_sparse.py` — sparse solver correctness tests.
- `tests/test_duals.py` — dual potential / strong-duality / complementary-slackness tests for both paths.

**Modified files:**
- `src/cpp/bonneel_solver.cpp` — `solve_dense` returns `(G, u, v)`; new `solve_sparse` returns `(rows, cols, vals, u, v)`.
- `src/sparse_ot/emd.py` — POT-compatible signatures; `log` dict with `cost`/`u`/`v`/`warning`/`result_code`; `center_dual` gauge shift; dispatch on `issparse(M)`.
- `src/sparse_ot/__init__.py` — drop any LEMON/OR-Tools re-exports.
- `tests/test_emd.py` — extend to assert on duals.
- `benchmarks/bench_solvers.py` — Bonneel-only.
- `CMakeLists.txt` — drop `_lemon` target.
- `pyproject.toml` — drop `ortools` dep.

**Deleted files:**
- `src/cpp/lemon_solver.cpp`, `src/cpp/lemon/` (entire vendored tree).
- `src/sparse_ot/ortools_solver.py`.
- `src/sparse_ot/routing.py`.
- `tests/test_lemon_accuracy.py`, `tests/test_ortools.py`, `tests/test_routing.py`.
- `src/sparse_ot/_ext/_lemon*.so` (build artifact; will regenerate).
- `benchmarks/results/routing_thresholds.json`.

---

## Task 1: Expose dual potentials from `solve_dense`

**Files:**
- Modify: `src/cpp/bonneel_solver.cpp`
- Modify: `src/sparse_ot/emd.py`
- Create: `tests/test_duals.py`

- [ ] **Step 1.1: Write failing duality test for the dense path**

Create `tests/test_duals.py`:

```python
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
    # u[i] + v[j] <= M[i,j] for all i,j
    slack = M - (u[:, None] + v[None, :])
    assert slack.min() > -1e-9


def test_dense_complementary_slackness():
    a, b, M = _problem(7, 9)
    G, info = sparse_ot.emd(a, b, M, log=True)
    u, v = info["u"], info["v"]
    # G[i,j] > 0 => u[i] + v[j] == M[i,j]
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
    # Gauge shift preserves u+v on the joint grid
    s_t = info_t["u"][:, None] + info_t["v"][None, :]
    s_f = info_f["u"][:, None] + info_f["v"][None, :]
    np.testing.assert_allclose(s_t, s_f, atol=1e-9)
```

- [ ] **Step 1.2: Run test, confirm it fails**

Run: `pytest tests/test_duals.py -v`
Expected: FAIL — `emd()` currently returns `(G, {})` with empty info dict.

- [ ] **Step 1.3: Modify `solve_dense` in `src/cpp/bonneel_solver.cpp` to return duals**

Replace the entire file with:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include "bonneel/full_bipartitegraph.h"
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;
using namespace lemon;
typedef FullBipartiteDigraph Digraph;
DIGRAPH_TYPEDEFS(Digraph);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
solve_dense(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> M,
    int numItermax
) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    auto M_buf = M.request();

    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const double* ap = static_cast<const double*>(a_buf.ptr);
    const double* bp = static_cast<const double*>(b_buf.ptr);
    const double* Mp = static_cast<const double*>(M_buf.ptr);

    py::array_t<double> G({n, m});
    py::array_t<double> u(n);
    py::array_t<double> v(m);
    double* Gp = static_cast<double*>(G.request().ptr);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (n == 0 || m == 0) {
        std::fill(Gp, Gp + n * m, 0.0);
        std::fill(up, up + n, 0.0);
        std::fill(vp, vp + m, 0.0);
        return std::make_tuple(G, u, v);
    }

    Digraph di(n, m);
    NetworkSimplexSimple<Digraph, double, double, int64_t> net(
        di, true, n + m, (int64_t)n * m, (size_t)numItermax
    );

    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            net.setCost(di.arcFromId((int64_t)i * m + j), Mp[i * m + j]);

    net.run();

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Gp[i * m + j] = net.flow(di.arcFromId((int64_t)i * m + j));

    // Dual potentials. POT convention u + v <= M, so flip pi sign on sources.
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));

    return std::make_tuple(G, u, v);
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for balanced OT";
    m.def("solve_dense", &solve_dense,
          py::arg("a"), py::arg("b"), py::arg("M"),
          py::arg("numItermax") = 100000,
          "Solve balanced OT. Returns (G, u, v).");
}
```

- [ ] **Step 1.4: Modify `src/sparse_ot/emd.py` to unpack duals and populate log dict**

Replace the body of `emd()` with the version that handles duals (keep `emd2` for now, we'll touch it in Task 7). For now only fix the dense branch — the sparse branch still goes through LEMON until Task 4.

Edit `src/sparse_ot/emd.py`: find the block

```python
    if selected == 'bonneel':
        M_dense = np.asarray(M, dtype=np.float64, order='C')
        G = _bonneel.solve_dense(a, b, M_dense, numItermax)
        if log:
            return G, {}
        return G
```

Replace with:

```python
    if selected == 'bonneel':
        M_dense = np.ascontiguousarray(M, dtype=np.float64)
        G, u, v = _bonneel.solve_dense(a, b, M_dense, numItermax)
        if center_dual:
            shift = float(u.mean())
            u = u - shift
            v = v + shift
        if log:
            cost = float(np.sum(G * M_dense))
            return G, {"cost": cost, "u": u, "v": v,
                       "warning": None, "result_code": 1}
        return G
```

- [ ] **Step 1.5: Rebuild and run tests**

Run: `pip install -e . --no-build-isolation -q && pytest tests/test_duals.py -v`
Expected: PASS — all six duality tests pass.

- [ ] **Step 1.6: Run full test suite to confirm no regressions**

Run: `pytest tests/ -v --ignore=tests/test_lemon_accuracy.py --ignore=tests/test_ortools.py`
Expected: PASS (LEMON/OR-Tools tests excluded — they'll be deleted in Tasks 5–6).

- [ ] **Step 1.7: Commit**

```bash
git add src/cpp/bonneel_solver.cpp src/sparse_ot/emd.py tests/test_duals.py
git commit -m "feat(bonneel): expose dual potentials from solve_dense"
```

---

## Task 2: Add the sparse bipartite digraph header

**Files:**
- Create: `src/cpp/bonneel/bipartite_sparse_digraph.h`

This header is exercised through the C++ solver in Task 3; no standalone test. Keeping it as its own task makes review clean.

- [ ] **Step 2.1: Create the header**

Create `src/cpp/bonneel/bipartite_sparse_digraph.h`:

```cpp
#ifndef BIPARTITE_SPARSE_DIGRAPH_H
#define BIPARTITE_SPARSE_DIGRAPH_H

#include <cstdint>
#include <algorithm>

namespace lemon {

  // CSR-backed bipartite digraph view conforming to the subset of the LEMON
  // Digraph concept used by NetworkSimplexSimple::init() and its pivot loop.
  //
  // Nodes 0..n1-1 are sources; nodes n1..n1+n2-1 are sinks.
  // Arc id a in [0, k) corresponds to CSR entry a: target = col_idx[a] + n1,
  // source = the unique i such that row_ptr[i] <= a < row_ptr[i+1].
  class BipartiteSparseDigraph {
  public:
    typedef int     Node;
    typedef int64_t Arc;

    BipartiteSparseDigraph(int n1, int n2,
                           const int* row_ptr, const int* col_idx, int64_t k)
      : _n1(n1), _n2(n2), _node_num(n1 + n2),
        _arc_num(k), _row_ptr(row_ptr), _col_idx(col_idx) {}

    int     nodeNum()  const { return _node_num; }
    int64_t arcNum()   const { return _arc_num; }
    int     maxNodeId() const { return _node_num - 1; }
    int64_t maxArcId()  const { return _arc_num - 1; }

    Node operator()(int ix) const { return Node(ix); }
    static int index(const Node& n) { return n; }

    Node source(Arc a) const {
        // Binary search: largest i with row_ptr[i] <= a.
        int lo = 0, hi = _n1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (_row_ptr[mid] <= a) lo = mid; else hi = mid - 1;
        }
        return Node(lo);
    }
    Node target(Arc a) const { return Node(_col_idx[a] + _n1); }

    static int    id(Node n) { return n; }
    static int64_t id(Arc a) { return a; }
    static Node nodeFromId(int i)    { return Node(i); }
    static Arc  arcFromId(int64_t i) { return Arc(i); }

    void first(Node& n) const { n = _node_num - 1; }
    static void next(Node& n) { --n; }

    void first(Arc& a) const { a = _arc_num - 1; }
    static void next(Arc& a) { --a; }

  private:
    int _n1, _n2, _node_num;
    int64_t _arc_num;
    const int* _row_ptr;
    const int* _col_idx;
  };

} // namespace lemon

#endif // BIPARTITE_SPARSE_DIGRAPH_H
```

- [ ] **Step 2.2: Sanity-check it compiles via the existing build**

The header is unused yet, so just confirm it does not break the build.

Run: `pip install -e . --no-build-isolation -q`
Expected: build succeeds.

- [ ] **Step 2.3: Commit**

```bash
git add src/cpp/bonneel/bipartite_sparse_digraph.h
git commit -m "feat(bonneel): add BipartiteSparseDigraph CSR-backed view"
```

---

## Task 3: Add `solve_sparse` C++ entry point

**Files:**
- Modify: `src/cpp/bonneel_solver.cpp`
- Create: `tests/test_bonneel_sparse.py`

- [ ] **Step 3.1: Write failing test that calls `_bonneel.solve_sparse` directly**

Create `tests/test_bonneel_sparse.py`:

```python
import numpy as np
import pytest
import scipy.sparse

from sparse_ot._ext import _bonneel


def _csr(M):
    sp = scipy.sparse.csr_matrix(M)
    return (sp.indptr.astype(np.int32),
            sp.indices.astype(np.int32),
            sp.data.astype(np.float64),
            sp.shape[0], sp.shape[1])


def test_solve_sparse_full_support_matches_dense():
    rng = np.random.default_rng(0)
    n, m = 6, 8
    a = rng.dirichlet(np.ones(n)); a = a / a.sum()
    b = rng.dirichlet(np.ones(m)); b = b / b.sum()
    M = rng.uniform(0.0, 1.0, size=(n, m))

    row_ptr, col_idx, costs, _, _ = _csr(M)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    G_sp = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    G_d, u_d, v_d = _bonneel.solve_dense(a, b, M, 100000)

    np.testing.assert_allclose(np.sum(G_sp * M), np.sum(G_d * M), atol=1e-9)
    np.testing.assert_allclose(G_sp.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G_sp.sum(axis=0), b, atol=1e-9)


def test_solve_sparse_returns_duals():
    rng = np.random.default_rng(1)
    n, m = 5, 5
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))
    row_ptr, col_idx, costs, _, _ = _csr(M)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    assert u.shape == (n,)
    assert v.shape == (m,)
    G = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    primal = float(np.sum(G * M))
    dual = float(a @ u + b @ v)
    assert abs(primal - dual) < 1e-9


def test_solve_sparse_knn_support():
    rng = np.random.default_rng(2)
    n, m = 10, 10
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))

    # Keep 4 cheapest entries per row
    k = 4
    keep = np.argsort(M, axis=1)[:, :k]
    mask = np.zeros_like(M, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    M_sparse = np.where(mask, M, 0.0)
    sp = scipy.sparse.csr_matrix(M_sparse)
    # Drop zeros so the support is exactly k per row
    sp.eliminate_zeros()
    row_ptr = sp.indptr.astype(np.int32)
    col_idx = sp.indices.astype(np.int32)
    costs = sp.data.astype(np.float64)

    rows, cols, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 100000
    )
    G = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, m)
    ).toarray()
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/test_bonneel_sparse.py -v`
Expected: FAIL — `_bonneel.solve_sparse` does not exist.

- [ ] **Step 3.3: Add `solve_sparse` to `src/cpp/bonneel_solver.cpp`**

Replace the file with:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include "bonneel/full_bipartitegraph.h"
#include "bonneel/bipartite_sparse_digraph.h"
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;
using namespace lemon;

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
solve_dense(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> M,
    int numItermax
) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    auto M_buf = M.request();
    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const double* ap = static_cast<const double*>(a_buf.ptr);
    const double* bp = static_cast<const double*>(b_buf.ptr);
    const double* Mp = static_cast<const double*>(M_buf.ptr);

    py::array_t<double> G({n, m});
    py::array_t<double> u(n);
    py::array_t<double> v(m);
    double* Gp = static_cast<double*>(G.request().ptr);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (n == 0 || m == 0) {
        std::fill(Gp, Gp + n * m, 0.0);
        std::fill(up, up + n, 0.0);
        std::fill(vp, vp + m, 0.0);
        return std::make_tuple(G, u, v);
    }

    FullBipartiteDigraph di(n, m);
    NetworkSimplexSimple<FullBipartiteDigraph, double, double, int64_t> net(
        di, true, n + m, (int64_t)n * m, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            net.setCost(di.arcFromId((int64_t)i * m + j), Mp[i * m + j]);
    net.run();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Gp[i * m + j] = net.flow(di.arcFromId((int64_t)i * m + j));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(G, u, v);
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    int numItermax
) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_idx.request();
    auto c_buf  = costs.request();

    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const int64_t k = static_cast<int64_t>(c_buf.size);
    const double* ap  = static_cast<const double*>(a_buf.ptr);
    const double* bp  = static_cast<const double*>(b_buf.ptr);
    const int*    rp  = static_cast<const int*>(rp_buf.ptr);
    const int*    ci  = static_cast<const int*>(ci_buf.ptr);
    const double* cp  = static_cast<const double*>(c_buf.ptr);

    py::array_t<double> u(n);
    py::array_t<double> v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int>    rows(0);
        py::array_t<int>    cols(0);
        py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0);
        std::fill(vp, vp + m, 0.0);
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

    net.run();

    // Collect nonzero flows
    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k);
    out_cols.reserve(k);
    out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }

    py::array_t<int>    rows_out(out_rows.size());
    py::array_t<int>    cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(),
                out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(),
                out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(),
                out_vals.size() * sizeof(double));

    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));

    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for balanced OT (dense and sparse)";
    m.def("solve_dense",  &solve_dense,
          py::arg("a"), py::arg("b"), py::arg("M"),
          py::arg("numItermax") = 100000,
          "Solve dense balanced OT. Returns (G, u, v).");
    m.def("solve_sparse", &solve_sparse,
          py::arg("a"), py::arg("b"),
          py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
          py::arg("numItermax") = 100000,
          "Solve sparse balanced OT on a CSR support. "
          "Returns (rows, cols, vals, u, v).");
}
```

- [ ] **Step 3.4: Build and run tests**

Run: `pip install -e . --no-build-isolation -q && pytest tests/test_bonneel_sparse.py -v`
Expected: PASS — all three tests green.

- [ ] **Step 3.5: Commit**

```bash
git add src/cpp/bonneel_solver.cpp tests/test_bonneel_sparse.py
git commit -m "feat(bonneel): add solve_sparse C++ entry on sparse digraph"
```

---

## Task 4: Switch sparse dispatch in `emd.py` from LEMON to Bonneel

**Files:**
- Modify: `src/sparse_ot/emd.py`
- Modify: `tests/test_emd.py`

- [ ] **Step 4.1: Add a failing test for sparse-input + duals through the Python API**

Append to `tests/test_duals.py`:

```python
def test_sparse_log_dict_and_duality():
    rng = np.random.default_rng(3)
    n, m = 8, 8
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M_d = rng.uniform(0.0, 1.0, size=(n, m))
    M_sp = scipy.sparse.csr_matrix(M_d)

    G, info = sparse_ot.emd(a, b, M_sp, log=True)
    assert scipy.sparse.issparse(G)
    assert set(info.keys()) >= {"cost", "u", "v", "warning", "result_code"}

    primal = float(G.multiply(M_d).sum())
    dual = float(a @ info["u"] + b @ info["v"])
    assert abs(primal - dual) < 1e-9
```

- [ ] **Step 4.2: Run, confirm failure**

Run: `pytest tests/test_duals.py::test_sparse_log_dict_and_duality -v`
Expected: FAIL — sparse path still routes through LEMON which returns no duals.

- [ ] **Step 4.3: Rewrite `src/sparse_ot/emd.py`**

Replace the entire file with:

```python
import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.feasibility import check_feasibility


def emd(a, b, M, numItermax=100000, log=False, center_dual=True):
    """Transport plan between distributions a and b with cost matrix M.

    POT-compatible: drop-in for ``ot.emd``. Dense numpy ``M`` returns a dense
    ndarray; ``scipy.sparse`` ``M`` returns a CSR.

    Parameters
    ----------
    a, b : array-like
    M    : ndarray (n, m) or scipy.sparse (n, m)
    numItermax  : int
    log         : bool — if True, return ``(G, info)`` with keys
                  ``cost, u, v, warning, result_code``.
    center_dual : bool — apply gauge shift ``u -= u.mean(); v += u.mean()``.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / a.sum()
    b = b / b.sum()

    if scipy.sparse.issparse(M):
        row_ptr, col_idx, costs, n, m, _ = to_csr(M, 0.0)
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        check_feasibility(a, b, row_ptr, col_idx)
        rows, cols, vals, u, v = _bonneel.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
        G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
        M_for_cost = M  # sparse
    else:
        M_dense = np.ascontiguousarray(M, dtype=np.float64)
        n, m = M_dense.shape
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        G, u, v = _bonneel.solve_dense(a, b, M_dense, numItermax)
        M_for_cost = M_dense

    if center_dual:
        shift = float(u.mean())
        u = u - shift
        v = v + shift

    if log:
        if scipy.sparse.issparse(G):
            cost = float(G.multiply(M_for_cost).sum())
        else:
            cost = float(np.sum(G * M_for_cost))
        return G, {"cost": cost, "u": u, "v": v,
                   "warning": None, "result_code": 1}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False):
    """POT-compatible ``ot.emd2``."""
    G, info = emd(a, b, M, numItermax=numItermax, log=True)
    cost = info["cost"]
    if return_matrix:
        info = {**info, "G": G}
        return (cost, info) if log else (cost, G)
    return (cost, info) if log else cost
```

- [ ] **Step 4.4: Run the new test and the full suite**

Run: `pytest tests/test_duals.py tests/test_bonneel_sparse.py tests/test_emd.py -v`
Expected: PASS — sparse duality test green; existing emd tests still green.

- [ ] **Step 4.5: Commit**

```bash
git add src/sparse_ot/emd.py tests/test_duals.py
git commit -m "feat(emd): route sparse M through Bonneel; POT-compat log dict"
```

---

## Task 5: Delete LEMON CostScaling

**Files:**
- Delete: `src/cpp/lemon_solver.cpp`, `src/cpp/lemon/` (entire directory)
- Delete: `tests/test_lemon_accuracy.py`
- Modify: `CMakeLists.txt` — remove `_lemon` target
- Delete: `src/sparse_ot/_ext/_lemon*.so` (build artifact)

- [ ] **Step 5.1: Confirm no remaining Python imports of `_lemon`**

Run: `grep -rn "_lemon\b\|from sparse_ot._ext import _lemon\|_ext._lemon" src/ tests/ benchmarks/ --include='*.py'`
Expected: results only in `benchmarks/` and `tests/test_lemon_accuracy.py` (both will be handled).

- [ ] **Step 5.2: Delete files and CMake target**

```bash
git rm tests/test_lemon_accuracy.py
git rm -r src/cpp/lemon src/cpp/lemon_solver.cpp
rm -f src/sparse_ot/_ext/_lemon*.so
```

Edit `CMakeLists.txt` to remove the `_lemon` block. The remaining file should be:

```cmake
cmake_minimum_required(VERSION 3.18)
project(sparse_ot_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_bonneel src/cpp/bonneel_solver.cpp)
target_include_directories(_bonneel PRIVATE src/cpp)
target_compile_options(_bonneel PRIVATE -O3)
target_compile_definitions(_bonneel PRIVATE NOOMP)
install(TARGETS _bonneel DESTINATION sparse_ot/_ext)
```

- [ ] **Step 5.3: Rebuild and run all remaining tests**

Run: `pip install -e . --no-build-isolation -q && pytest tests/ --ignore=tests/test_ortools.py --ignore=tests/test_routing.py --ignore=tests/test_benchmarks.py -v`
Expected: PASS.

- [ ] **Step 5.4: Commit**

```bash
git add -A
git commit -m "chore: remove LEMON CostScaling solver and vendored headers"
```

---

## Task 6: Delete OR-Tools

**Files:**
- Delete: `src/sparse_ot/ortools_solver.py`
- Delete: `tests/test_ortools.py`
- Modify: `pyproject.toml` — drop `ortools` dependency / extras

- [ ] **Step 6.1: Confirm no remaining imports of `ortools_solver` outside the files we delete**

Run: `grep -rn "ortools_solver\|from sparse_ot.ortools_solver\|ortools_cost_scale" src/ tests/ benchmarks/ --include='*.py'`
Expected: results only in `tests/test_ortools.py`, `benchmarks/`, and any stale spot in `emd.py`. We've already removed `ortools_cost_scale` and the `ortools` branch in Task 4.

- [ ] **Step 6.2: Delete files**

```bash
git rm src/sparse_ot/ortools_solver.py tests/test_ortools.py
```

- [ ] **Step 6.3: Edit `pyproject.toml` to drop the `ortools` dependency**

Open `pyproject.toml`, find the dependency block, and remove the `ortools` entry (whether under `dependencies`, `[project.optional-dependencies]`, or any tool-specific section). After the edit, run:

Run: `grep -n "ortools" pyproject.toml`
Expected: no matches.

- [ ] **Step 6.4: Rebuild and test**

Run: `pip install -e . --no-build-isolation -q && pytest tests/ --ignore=tests/test_routing.py --ignore=tests/test_benchmarks.py -v`
Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add -A
git commit -m "chore: remove OR-Tools solver and dependency"
```

---

## Task 7: Delete `routing.py` and simplify dispatch

**Files:**
- Delete: `src/sparse_ot/routing.py`
- Delete: `tests/test_routing.py`
- Delete: `benchmarks/results/routing_thresholds.json`
- Modify: `src/sparse_ot/__init__.py` (if it re-exports anything routing-related)

`emd.py` already dispatches purely on `issparse(M)` after Task 4, so no Python code change is required here beyond deleting routing.

- [ ] **Step 7.1: Confirm no remaining imports of `routing`**

Run: `grep -rn "from sparse_ot.routing\|import routing\|select_solver" src/ tests/ benchmarks/ --include='*.py'`
Expected: no matches except inside the file we're deleting and possibly its test.

- [ ] **Step 7.2: Delete files**

```bash
git rm src/sparse_ot/routing.py tests/test_routing.py benchmarks/results/routing_thresholds.json
```

- [ ] **Step 7.3: Check `__init__.py`**

Read `src/sparse_ot/__init__.py`. If it imports anything from `routing`, delete that line. (Currently it only imports from `emd` and `feasibility`.)

- [ ] **Step 7.4: Test**

Run: `pytest tests/ --ignore=tests/test_benchmarks.py -v`
Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add -A
git commit -m "chore: delete routing module; dispatch is now issparse(M)"
```

---

## Task 8: Bonneel-only benchmarks

**Files:**
- Modify: `benchmarks/bench_solvers.py`
- Modify: `tests/test_benchmarks.py`

- [ ] **Step 8.1: Inspect current bench**

Run: `cat benchmarks/bench_solvers.py`
Note where LEMON, OR-Tools, and the densify-with-penalty branch are referenced.

- [ ] **Step 8.2: Strip LEMON / OR-Tools / densify branches**

Edit `benchmarks/bench_solvers.py`:
- Remove any import of `_lemon`, `ortools_solver`, or `select_solver`.
- Remove any benchmark cases that explicitly target `'lemon'` or `'ortools'`.
- Remove the densify-with-penalty special case for Bonneel on sparse M (Bonneel now handles sparse natively).
- The remaining bench should iterate over `(n, m, density)` configurations and time `sparse_ot.emd(a, b, M)` with both dense and sparse `M`.

Show the resulting file is import-clean:

Run: `python -c "import benchmarks.bench_solvers"`
Expected: imports without error.

- [ ] **Step 8.3: Update `tests/test_benchmarks.py`**

Open `tests/test_benchmarks.py`. Delete any assertion that mentions LEMON, OR-Tools, or routing thresholds. The remaining test should just call the bench entry point on a tiny configuration and assert it produces a result dict.

If the existing test is entirely about routing thresholds, delete the file:

```bash
git rm tests/test_benchmarks.py  # only if it is exclusively routing-focused
```

Otherwise edit it inline to remove the dead checks.

- [ ] **Step 8.4: Run**

Run: `pytest tests/ -v`
Expected: PASS.

- [ ] **Step 8.5: Commit**

```bash
git add -A
git commit -m "chore(bench): Bonneel-only benchmarks; drop densify workaround"
```

---

## Task 9: Memory smoke test

**Files:**
- Modify: `tests/test_bonneel_sparse.py`

This is the spec's promised memory claim, made executable.

- [ ] **Step 9.1: Append the memory smoke test**

Append to `tests/test_bonneel_sparse.py`:

```python
import resource
import sys


@pytest.mark.slow
def test_sparse_memory_scales_with_k():
    """10k x 10k with k=100k must fit comfortably under O(n*m) memory."""
    rng = np.random.default_rng(42)
    n = m = 10_000
    k_per_row = 10
    k = n * k_per_row

    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    a = a / a.sum()
    b = b / b.sum()

    # Pick k_per_row random columns per row, with random positive costs.
    cols = np.stack([
        rng.choice(m, size=k_per_row, replace=False) for _ in range(n)
    ], axis=0)
    cols.sort(axis=1)
    row_ptr = (np.arange(n + 1) * k_per_row).astype(np.int32)
    col_idx = cols.ravel().astype(np.int32)
    costs   = rng.uniform(0.0, 1.0, size=k).astype(np.float64)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows, cols_o, vals, u, v = _bonneel.solve_sparse(
        a, b, row_ptr, col_idx, costs, 1_000_000
    )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is KB on Linux, bytes on macOS. Normalize to MB.
    scale = 1024.0 if sys.platform == "darwin" else 1.0
    delta_mb = (rss_after - rss_before) * scale / (1024.0 * 1024.0)
    # Dense Bonneel would need ~800 MB for the cost matrix alone.
    # Sparse with k=100k should be well under 200 MB.
    assert delta_mb < 200, f"RSS grew by {delta_mb:.1f} MB, expected < 200 MB"
```

The test is marked `slow` so it stays off the default run; opt in via `pytest -m slow` or `pytest tests/test_bonneel_sparse.py::test_sparse_memory_scales_with_k -v`.

- [ ] **Step 9.2: Run explicitly**

Run: `pytest tests/test_bonneel_sparse.py::test_sparse_memory_scales_with_k -v`
Expected: PASS.

- [ ] **Step 9.3: Run full suite one final time**

Run: `pytest tests/ -v`
Expected: PASS.

- [ ] **Step 9.4: Commit**

```bash
git add tests/test_bonneel_sparse.py
git commit -m "test(bonneel): sparse memory stays under 200 MB at k=100k, n=10k"
```

---

## Done

After Task 9: the solver surface is {Bonneel-dense, Bonneel-sparse}, both return duals, the public API matches POT, and memory scales with `k` rather than `n·m`.
