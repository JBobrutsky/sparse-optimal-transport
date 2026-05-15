# Plan 1: Scaffold + Bonneel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a working `sparse-ot` package that exposes `emd()` and `emd2()` backed by Bonneel's network simplex, passing all correctness tests against POT on dense numpy inputs.

**Architecture:** pybind11 C++ extension wrapping Bonneel's `network_simplex_simple.h`, built via scikit-build-core. A thin Python layer in `emd.py` normalizes inputs and matches POT's API signature exactly. No routing, no sparse inputs, no LEMON — those come in Plan 2.

**Tech Stack:** Python 3.12 (uv), scikit-build-core, pybind11, CMake 3.18+, POT (dev/test only), pytest

---

## File Map

| File | Role |
|---|---|
| `pyproject.toml` | Package metadata, build system, deps |
| `CMakeLists.txt` | Builds the `_bonneel` pybind11 extension |
| `src/sparse_ot/__init__.py` | Exports `emd`, `emd2` |
| `src/sparse_ot/emd.py` | Public API — normalizes inputs, calls `_bonneel.solve_dense` |
| `src/sparse_ot/_ext/__init__.py` | Empty — marks `_ext` as a package |
| `src/cpp/bonneel/network_simplex_simple.h` | Vendored from Bonneel's repo |
| `src/cpp/bonneel_solver.cpp` | pybind11 wrapper exposing `solve_dense` |
| `tests/test_emd.py` | Correctness tests vs POT |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `CMakeLists.txt`
- Create: `src/sparse_ot/__init__.py`
- Create: `src/sparse_ot/_ext/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/sparse_ot/_ext src/cpp/bonneel tests
touch src/sparse_ot/_ext/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "pybind11>=2.12"]
build-backend = "scikit-build-core.build"

[project]
name = "sparse-ot"
version = "0.1.0"
description = "Sparse optimal transport with drop-in POT API"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
ortools = ["ortools>=9.8"]
dev = ["pytest>=7.0", "pot>=0.9", "pybind11>=2.12"]

[tool.scikit-build]
wheel.packages = ["src/sparse_ot"]
cmake.build-type = "Release"
```

- [ ] **Step 3: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(sparse_ot_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_bonneel src/cpp/bonneel_solver.cpp)
target_include_directories(_bonneel PRIVATE src/cpp)
target_compile_options(_bonneel PRIVATE -O3)
install(TARGETS _bonneel DESTINATION sparse_ot/_ext)
```

- [ ] **Step 4: Write stub `src/sparse_ot/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Create the uv virtual environment and install dev deps**

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install pot numpy scipy pytest pybind11 scikit-build-core
```

- [ ] **Step 6: Verify Python version**

```bash
python --version
```
Expected: `Python 3.12.x`

- [ ] **Step 7: Commit scaffold**

```bash
git add pyproject.toml CMakeLists.txt src/ tests/
git commit -m "feat: project scaffold — pyproject.toml, CMakeLists, package stub"
```

---

## Task 2: Write Failing Tests

**Files:**
- Create: `tests/test_emd.py`

- [ ] **Step 1: Write `tests/test_emd.py`**

```python
import numpy as np
import pytest
import ot

import sparse_ot


def _problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    M = rng.uniform(0.0, 1.0, size=(n, m))
    return a, b, M


def test_emd_shape():
    a, b, M = _problem(5, 7)
    G = sparse_ot.emd(a, b, M)
    assert G.shape == (5, 7)


def test_emd_row_marginals():
    a, b, M = _problem(8, 6)
    G = sparse_ot.emd(a, b, M)
    np.testing.assert_allclose(G.sum(axis=1), a, atol=1e-9)


def test_emd_col_marginals():
    a, b, M = _problem(8, 6)
    G = sparse_ot.emd(a, b, M)
    np.testing.assert_allclose(G.sum(axis=0), b, atol=1e-9)


def test_emd_matches_pot_rectangular():
    a, b, M = _problem(10, 12)
    G = sparse_ot.emd(a, b, M)
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd_matches_pot_square():
    a, b, M = _problem(20, 20)
    G = sparse_ot.emd(a, b, M)
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd2_matches_pot():
    a, b, M = _problem(15, 15)
    cost = sparse_ot.emd2(a, b, M)
    cost_ref = ot.emd2(a, b, M)
    assert abs(cost - cost_ref) / abs(cost_ref) < 1e-6


def test_emd2_consistent_with_emd():
    a, b, M = _problem(10, 10)
    G = sparse_ot.emd(a, b, M)
    cost_from_plan = float(np.sum(G * M))
    cost_from_emd2 = sparse_ot.emd2(a, b, M)
    assert abs(cost_from_plan - cost_from_emd2) < 1e-12


def test_emd_solver_override_bonneel():
    a, b, M = _problem(8, 8)
    G = sparse_ot.emd(a, b, M, solver="bonneel")
    G_ref = ot.emd(a, b, M)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_emd_accepts_cost_sparsity_threshold():
    # Threshold parameter is accepted without error in Plan 1.
    # Routing behavior based on threshold comes in Plan 2.
    a, b, M = _problem(5, 5)
    G = sparse_ot.emd(a, b, M, cost_sparsity_threshold=0.01)
    assert G.shape == (5, 5)


def test_emd2_return_matrix():
    a, b, M = _problem(5, 5)
    cost, G = sparse_ot.emd2(a, b, M, return_matrix=True)
    assert isinstance(cost, float)
    assert G.shape == (5, 5)


def test_emd_log():
    a, b, M = _problem(5, 5)
    G, log = sparse_ot.emd(a, b, M, log=True)
    assert G.shape == (5, 5)
    assert isinstance(log, dict)


def test_emd_nonnegative_transport():
    a, b, M = _problem(10, 10)
    G = sparse_ot.emd(a, b, M)
    assert np.all(G >= -1e-12)
```

- [ ] **Step 2: Run tests — confirm they fail with ImportError**

```bash
pytest tests/test_emd.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'emd' from 'sparse_ot'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_emd.py
git commit -m "test: add failing emd/emd2 correctness tests vs POT"
```

---

## Task 3: Vendor Bonneel's Header

**Files:**
- Create: `src/cpp/bonneel/network_simplex_simple.h`

- [ ] **Step 1: Download Bonneel's header**

```bash
curl -L -o src/cpp/bonneel/network_simplex_simple.h \
  https://raw.githubusercontent.com/nbonneel/network_simplex/master/network_simplex_simple.h
```

- [ ] **Step 2: Verify the header downloaded correctly and note the class interface**

```bash
head -80 src/cpp/bonneel/network_simplex_simple.h
```
Expected output: C++ header with copyright notice and the `NetworkSimplexSimple` template class definition. Note the exact template parameter order and method names — you will need them in Task 4.

The interface used in Task 4 assumes:
```
NetworkSimplexSimple<NodeId, Value, Cost, ArcId>
  .setNodeSupply(NodeId node, Value supply)
  .addArc(NodeId from, NodeId to)  → ArcId
  .setCost(ArcId arc, Cost cost)
  .run(ProblemType type, ArcId maxIter)  → ProblemType
  .flow(ArcId arc)  → Value
```
If the actual method names differ, update `bonneel_solver.cpp` accordingly before building.

- [ ] **Step 3: Write `src/cpp/bonneel/VENDORING.md`**

```markdown
# Bonneel Network Simplex

Source: https://github.com/nbonneel/network_simplex  
File: network_simplex_simple.h  
License: See header file  
Vendored: 2026-05-15  
Modifications: none
```

- [ ] **Step 4: Commit vendored header**

```bash
git add src/cpp/bonneel/
git commit -m "vendor: add Bonneel network_simplex_simple.h"
```

---

## Task 4: C++ pybind11 Extension

**Files:**
- Create: `src/cpp/bonneel_solver.cpp`

- [ ] **Step 1: Write `src/cpp/bonneel_solver.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;

using NodeId = long long int;
using Simplex = NetworkSimplexSimple<NodeId, double, double, NodeId>;

// Solve a balanced optimal transport problem using Bonneel's network simplex.
//
// a:         source weights, shape (n,), sums to 1.0
// b:         target weights, shape (m,), sums to 1.0
// M:         cost matrix, shape (n, m), row-major float64
// numItermax: iteration limit
//
// Returns: transport plan G, shape (n, m), float64
py::array_t<double> solve_dense(
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
    const double* ap = static_cast<double*>(a_buf.ptr);
    const double* bp = static_cast<double*>(b_buf.ptr);
    const double* Mp = static_cast<double*>(M_buf.ptr);

    // n source nodes + m sink nodes; n*m arcs
    Simplex net(
        static_cast<NodeId>(n + m),   // node count
        true,                          // use LIFO pricing strategy
        static_cast<NodeId>(n + m),   // node count again (some versions require this)
        static_cast<NodeId>(n * m)    // arc count estimate
    );

    // Set supply (positive) for source nodes and demand (negative) for sink nodes.
    for (int i = 0; i < n; i++)
        net.setNodeSupply(static_cast<NodeId>(i), ap[i]);
    for (int j = 0; j < m; j++)
        net.setNodeSupply(static_cast<NodeId>(n + j), -bp[j]);

    // Add all (source, sink) arcs and set their costs.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            NodeId arc = net.addArc(
                static_cast<NodeId>(i),
                static_cast<NodeId>(n + j)
            );
            net.setCost(arc, Mp[i * m + j]);
        }
    }

    net.run(Simplex::OPTIMAL, static_cast<NodeId>(numItermax));

    // Extract transport plan: arcs were added in row-major order, so arc k
    // corresponds to source i = k/m, sink j = k%m.
    py::array_t<double> G({n, m});
    auto G_buf = G.request();
    double* Gp = static_cast<double*>(G_buf.ptr);

    for (NodeId k = 0; k < static_cast<NodeId>(n * m); k++)
        Gp[k] = net.flow(k);

    return G;
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for dense balanced OT";
    m.def(
        "solve_dense", &solve_dense,
        py::arg("a"), py::arg("b"), py::arg("M"),
        py::arg("numItermax") = 100000,
        "Solve balanced OT. a and b must sum to 1.0. M is (n,m) float64 row-major."
    );
}
```

- [ ] **Step 2: Build the extension**

```bash
uv pip install --no-build-isolation -e .
```
Expected: build output ending in `Successfully installed sparse-ot-0.1.0`

If you see a compilation error about a missing method (e.g., `setNodeSupply` not found), open `src/cpp/bonneel/network_simplex_simple.h`, search for `supply` or `addNode`, and update the method name in `bonneel_solver.cpp` to match. Likewise for `addArc`, `setCost`, `flow`.

If the constructor signature differs, check the class definition near the top of the header for the constructor parameters and update accordingly.

- [ ] **Step 3: Verify the extension imports**

```bash
python -c "from sparse_ot._ext import _bonneel; print(_bonneel.__doc__)"
```
Expected: `Bonneel network simplex for dense balanced OT`

- [ ] **Step 4: Smoke-test the solver directly**

```bash
python - <<'EOF'
import numpy as np
from sparse_ot._ext import _bonneel

a = np.array([0.5, 0.5])
b = np.array([0.3, 0.7])
M = np.array([[0.0, 1.0], [1.0, 0.0]])
G = _bonneel.solve_dense(a, b, M, 100000)
print("G =", G)
# Expected: all mass goes via the zero-cost diagonal
# G[0,0] ≈ 0.3, G[0,1] ≈ 0.2, G[1,0] ≈ 0.0, G[1,1] ≈ 0.5
print("row sums:", G.sum(axis=1))   # should be [0.5, 0.5]
print("col sums:", G.sum(axis=0))   # should be [0.3, 0.7]
EOF
```
Expected: row sums ≈ [0.5, 0.5], col sums ≈ [0.3, 0.7], cost-minimizing assignment used.

- [ ] **Step 5: Commit**

```bash
git add src/cpp/bonneel_solver.cpp
git commit -m "feat: pybind11 wrapper for Bonneel network simplex (_bonneel extension)"
```

---

## Task 5: Python API

**Files:**
- Create: `src/sparse_ot/emd.py`
- Modify: `src/sparse_ot/__init__.py`

- [ ] **Step 1: Write `src/sparse_ot/emd.py`**

```python
import numpy as np

from sparse_ot._ext import _bonneel


def emd(a, b, M, numItermax=100000, log=False, center_dual=True,
        cost_sparsity_threshold=0.0, solver=None):
    """Transport plan between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd(). Returns a dense numpy array in Plan 1.
    Sparse inputs and LEMON/OR-Tools routing are added in Plans 2 and 3.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like, shape (n, m)
    numItermax : int
        Max iterations for the solver.
    log : bool
        If True, return (G, log_dict).
    center_dual : bool
        Accepted for API compatibility; not used in Plan 1.
    cost_sparsity_threshold : float
        Accepted for API compatibility; routing uses this in Plan 2.
    solver : str or None
        'bonneel', or None (auto). 'lemon' and 'ortools' added in Plans 2-3.

    Returns
    -------
    G : ndarray, shape (n, m)
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64, order="C")

    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    if M.shape != (len(a), len(b)):
        raise ValueError(
            f"M must have shape ({len(a)}, {len(b)}), got {M.shape}"
        )
    if solver not in (None, "bonneel"):
        raise ValueError(
            f"solver={solver!r} not available in Plan 1. "
            "Use None or 'bonneel'. 'lemon' and 'ortools' come in Plans 2-3."
        )

    # Bonneel requires exactly balanced supply/demand.
    a = a / a.sum()
    b = b / b.sum()

    G = _bonneel.solve_dense(a, b, M, numItermax)

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
    M : array-like, shape (n, m)
    numItermax : int
    log : bool
        If True, return (cost, log_dict) or (cost, G, log_dict) with return_matrix.
    return_matrix : bool
        If True, also return the transport plan G.
    cost_sparsity_threshold : float
    solver : str or None

    Returns
    -------
    cost : float
    G : ndarray, shape (n, m)  — only if return_matrix=True
    """
    G = emd(a, b, M, numItermax=numItermax, log=False,
            cost_sparsity_threshold=cost_sparsity_threshold, solver=solver)

    M_n = np.asarray(M, dtype=np.float64)
    cost = float(np.sum(G * M_n))

    if return_matrix:
        if log:
            return cost, G, {}
        return cost, G
    if log:
        return cost, {}
    return cost
```

- [ ] **Step 2: Update `src/sparse_ot/__init__.py`**

```python
from .emd import emd, emd2

__version__ = "0.1.0"
__all__ = ["emd", "emd2"]
```

- [ ] **Step 3: Reinstall so the new Python files are picked up**

```bash
uv pip install --no-build-isolation -e .
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/test_emd.py -v
```
Expected: all 12 tests pass.

If `test_emd_matches_pot_rectangular` or `test_emd_matches_pot_square` fail with a tolerance error, the issue is likely a sign convention difference in how Bonneel returns flows (e.g., the arc ID ordering doesn't match the i*m+j layout assumed in Task 4). Fix by printing `G` and `G_ref` side by side:

```bash
python - <<'EOF'
import numpy as np, ot, sparse_ot
rng = np.random.default_rng(0)
a = rng.dirichlet(np.ones(3))
b = rng.dirichlet(np.ones(3))
M = rng.uniform(0, 1, (3, 3))
print("ours:\n", sparse_ot.emd(a, b, M))
print("POT:\n", ot.emd(a, b, M))
EOF
```

If the matrices are permuted, the arc extraction in `bonneel_solver.cpp` (the `net.flow(k)` loop) needs to match the order in which arcs were added. Verify by changing the loop to use the arc ID returned by `addArc`:

```cpp
// Alternative extraction if arc IDs are not sequential from 0:
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
        Gp[i * m + j] = net.flow(net.arcFromId(i * m + j));
```
Rebuild with `uv pip install --no-build-isolation -e .` and re-run tests.

- [ ] **Step 5: Commit**

```bash
git add src/sparse_ot/emd.py src/sparse_ot/__init__.py
git commit -m "feat: emd() and emd2() Python API backed by Bonneel network simplex"
```

---

## Task 6: Final Verification and Commit

- [ ] **Step 1: Run full test suite clean**

```bash
pytest tests/test_emd.py -v --tb=short
```
Expected: 12 passed, 0 failed, 0 errors.

- [ ] **Step 2: Verify the package installs cleanly from scratch into a fresh venv**

```bash
uv venv .venv-check --python 3.12
source .venv-check/bin/activate
uv pip install pot pytest pybind11 scikit-build-core numpy scipy
uv pip install --no-build-isolation .
pytest tests/test_emd.py -v
deactivate
rm -rf .venv-check
source .venv/bin/activate
```
Expected: all 12 tests pass in the fresh environment.

- [ ] **Step 3: Commit final state**

```bash
git add -A
git status  # verify no untracked files you didn't intend to add
git commit -m "feat: Plan 1 complete — sparse-ot with Bonneel solver, all tests pass"
```

---

## Troubleshooting Reference

**Build fails: `setNodeSupply` not found**
Open `src/cpp/bonneel/network_simplex_simple.h` and search for `supply`. The method may be named `addNode(id, supply)` or `nodeSupply`. Replace all `setNodeSupply` calls in `bonneel_solver.cpp` with the actual name.

**Build fails: constructor argument count mismatch**
Search for `NetworkSimplexSimple(` in the header to find the constructor. Match the argument count and types in `bonneel_solver.cpp`.

**`net.flow(k)` returns wrong values**
The arc returned by `addArc` may not be an integer ID starting at 0. Replace the extraction loop with:
```cpp
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
        // arc_ids were stored when addArc was called — store them in a vector
    }
```
Store `addArc` return values in `std::vector<NodeId> arc_ids(n*m)` during arc creation, then use `net.flow(arc_ids[i*m+j])` during extraction.

**`test_emd_matches_pot` fails with atol=1e-6**
Both solvers should find the exact same optimal solution for small problems. If they differ by more than 1e-6, check that `a` and `b` sum to exactly 1.0 before being passed to C++ (the normalization in `emd.py` handles this), and that the cost matrix has no NaN or Inf values.
