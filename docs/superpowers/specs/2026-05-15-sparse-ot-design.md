# sparse-ot Design Spec

**Date:** 2026-05-15  
**Status:** Approved

## Overview

`sparse-ot` is a Python package published to PyPI that provides drop-in replacements for POT's `emd()` and `emd2()` functions, optimized for heavily sparse bipartite graphs. It wraps two C++ solvers — Bonneel's network simplex (efficient for dense/near-dense cost matrices) and a float64-patched LEMON CostScaling (efficient for sparse cost matrices) — and routes between them automatically based on empirically derived sparsity thresholds. The package is structured to facilitate future merge into the official POT project.

**Reference use case:** restricted optimal transport between two fully dense 3D images of size 256×256×256 (n = m = 256³ ≈ 16.7M). In the dense case the cost matrix has 256⁶ ≈ 281 trillion entries. In the reference sparse case only certain transport paths are permitted (e.g. nearby voxels), giving 16 × 2 × 256³ ≈ 536M non-zero cost entries (~32 allowed neighbors per voxel) and a sparsity ratio of ~2×10⁻⁶. The distributions themselves are always treated as fully dense; all sparsity is in the cost matrix (the bipartite graph structure).

---

## 1. Repository Layout

```
sparse-optimal-transport/
├── src/
│   ├── sparse_ot/
│   │   ├── __init__.py              # exposes emd, emd2, set_backend
│   │   ├── emd.py                   # public API
│   │   ├── backend.py               # math backend abstraction (mirrors POT's)
│   │   ├── sparse_utils.py          # sparsity detection, thresholding, format conversion
│   │   ├── routing.py               # solver dispatch
│   │   └── _ext/                    # compiled pybind11 modules (build artifacts)
│   └── cpp/
│       ├── bonneel/
│       │   └── network_simplex_simple.h   # vendored from github.com/nbonneel/network_simplex
│       ├── lemon/                         # vendored LEMON headers (CostScaling subset, ~15 headers)
│       ├── bonneel_solver.cpp             # pybind11 wrapper for Bonneel
│       └── lemon_solver.cpp              # pybind11 wrapper for LEMON CostScaling
├── benchmarks/
│   ├── bench_solvers.py             # efficiency + accuracy sweep
│   ├── generate_report.py           # publication-quality figures + JSON results
│   └── results/                     # committed benchmark outputs and figures
├── tests/
│   ├── test_emd.py                  # correctness vs POT
│   ├── test_sparse_utils.py         # format conversion
│   ├── test_routing.py              # solver dispatch logic
│   └── test_lemon_accuracy.py       # float64 patch validation
├── docs/superpowers/specs/
├── pyproject.toml                   # scikit-build-core + package metadata
├── CMakeLists.txt                   # builds both C++ extensions
└── README.md                        # benchmarks, routing explanation (Bonneel-style)
```

---

## 2. Python API

The public API is a strict superset of POT's `emd` and `emd2` signatures. All POT-compatible call sites work unchanged.

```python
import sparse_ot as sot

# Transport plan — returns sparse or dense matrix depending on solver path
G = sot.emd(a, b, M, numItermax=100000, log=False, center_dual=True,
            cost_sparsity_threshold=0.0, solver=None)

# Transport cost — returns scalar
cost = sot.emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
                cost_sparsity_threshold=0.0, solver=None)
```

**Extensions beyond POT:**
- `cost_sparsity_threshold` (float, default 0.0): when `M` is a dense array, values with `|M[i,j]| <= cost_sparsity_threshold` are treated as absent edges. Exact zeros are always treated as absent regardless of this parameter.
- `solver` (str or None): `'bonneel'`, `'lemon'`, or `None` (auto-route). Bypasses routing when set.

**Output format:**
- Bonneel path: numpy float64 array (dense), matching POT's output exactly
- LEMON path: scipy CSR sparse matrix by default; if input backend is PyTorch, returns torch sparse COO

**`emd2` implementation:** calls `emd` internally, then contracts the transport plan against `M` to produce the scalar cost. Same as POT.

---

## 3. Data Flow

```
User calls emd(a, b, M)
    │
    ├─ backend.py
    │   detect backend (numpy / scipy sparse / torch dense / torch sparse)
    │   normalize a, b to sum-1 float64 numpy arrays
    │
    ├─ sparse_utils.py
    │   dense M      → apply cost_sparsity_threshold → internal CSR (float64 costs, int32 indices)
    │   scipy sparse → convert to CSR directly
    │   torch sparse → extract indices/values → CSR
    │   compute: nnz, sparsity_ratio = nnz / (n * m)
    │
    ├─ routing.py
    │   select_solver(n, m, sparsity_ratio, solver_override)
    │   → 'bonneel'  if sparsity_ratio > _threshold(n, m)
    │   → 'lemon'    otherwise
    │
    ├─ C++ extension
    │   Bonneel: (a, b, M_dense float64) → dense transport matrix float64
    │   LEMON:   (a, b, row_ptr, col_idx, costs) → COO (i, j, val) float64
    │
    └─ backend.py
        wrap result in output format matching input backend
        LEMON result always assembled as scipy CSR (or torch sparse COO)
```

---

## 4. C++ Extension Layer

### Bonneel solver (`bonneel_solver.cpp`)

Thin pybind11 wrapper over `network_simplex_simple.h`. Costs and transport values are float64 natively — no scaling required. Exposed interface:

```cpp
// sparse_ot._ext.bonneel
py::array_t<double> solve_dense(
    py::array_t<double> a,    // (n,)
    py::array_t<double> b,    // (m,)
    py::array_t<double> M,    // (n, m) row-major
    int numItermax
);
```

### LEMON solver (`lemon_solver.cpp`)

LEMON's `CostScaling` is vendored and patched for float64 support. The single required change is the termination criterion: `epsilon < 1` (integer assumption) is replaced with `epsilon < _tolerance * _initial_max_cost` (relative floating-point condition, `_tolerance = 1e-9`). The cost type template parameter is instantiated as `double`. Exposed interface:

```cpp
// sparse_ot._ext.lemon
std::tuple<py::array_t<int32_t>,
           py::array_t<int32_t>,
           py::array_t<double>>
solve_sparse(
    py::array_t<double>  a,        // (n,)
    py::array_t<double>  b,        // (m,)
    py::array_t<int32_t> row_ptr,  // (n+1,) CSR row pointer
    py::array_t<int32_t> col_idx,  // (nnz,) CSR column indices
    py::array_t<double>  costs,    // (nnz,) CSR edge costs
    int numItermax
);
// returns COO: (row_indices, col_indices, values)
```

The COO triplets returned by the C++ layer are assembled into scipy CSR (or torch sparse COO) in `backend.py` before being returned to the caller. The C++ layer has no dependency on scipy or torch.

**LEMON vendoring:** ~15 headers from the CostScaling dependency tree. BSD-licensed. The float64 patch is isolated to the termination condition and documented in `src/cpp/lemon/PATCHES.md`.

---

## 5. Routing Logic

```python
# routing.py
def select_solver(n: int, m: int, nnz: int, solver: str | None = None) -> str:
    if solver is not None:
        return solver
    k = nnz / n   # average neighbors per source node
    return 'bonneel' if k > _k_threshold(n, m) else 'lemon'
```

`_k_threshold(n, m)` is a lookup into a table loaded from `benchmarks/results/routing_thresholds.json` at import time. The table encodes the empirically derived crossover number of neighbors per node — the point at which LEMON becomes faster than Bonneel — for each `(n, m)` combination in the benchmark sweep. Using `k` rather than raw sparsity ratio makes the threshold numerically stable across very large n (where sparsity ratios approach floating-point underflow).

**Before benchmarks are run**, the package ships with a conservative default (`sparsity > 0.01` → Bonneel) that is safe across all problem sizes.

**The threshold table is regenerated** by running `python benchmarks/generate_report.py` and committing the updated JSON. The README documents this process for users who want hardware-specific routing.

---

## 6. Benchmark Suite

**Problem generator:** both distributions `a` and `b` are always fully dense (all n bins have positive mass, drawn from a Dirichlet distribution). Sparsity is controlled exclusively by `k` — the number of allowed neighbors per source node — so `nnz = k × n`. Problems are structured to match the reference use case: a regular grid in 1D/2D/3D where each source node connects to its `k` nearest neighbors in the target grid.

### Efficiency sweep

```
n        ∈ {1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M}
k        ∈ {2, 8, 32, 128, 512, 2048, n/10, n}     # neighbors per node; k=n is the fully dense case
solvers  = ['bonneel', 'lemon', 'pot_reference']
metrics  = wall_time (median of 5 runs), peak_memory_mb, iterations
```

Note: `pot_reference` and `bonneel` are only run where `k × n` fits in memory as a dense matrix (n ≤ 64K or k = n ≤ 16K). Results: `benchmarks/results/efficiency.json`

### Accuracy sweep

For each `(n, k)` configuration where LEMON is the selected solver:
- **n ≤ 64K:** ground truth via POT's `emd2`; report relative cost error `|cost_lemon - cost_pot| / cost_pot` and primal feasibility `‖T @ 1 − a‖∞`, `‖Tᵀ @ 1 − b‖∞`
- **n > 64K:** POT cannot run at this scale; report primal feasibility only (no cost error metric)

Results: `benchmarks/results/accuracy.json`

### Report figures (publication-quality, PDF + PNG)

- **Heatmap:** wall time ratio (LEMON/Bonneel) over the (n, k) grid; crossover contour marks the routing threshold
- **Line plots:** wall time vs. n at fixed k values, one line per solver
- **Accuracy plot:** relative cost error and feasibility residual vs. k, with confidence bands across random problem instances
- **Routing threshold derivation:** annotated crossover contour used to generate `routing_thresholds.json`

Figures saved to `benchmarks/results/figures/`. Key figures embedded in README.

---

## 7. Testing Strategy

| Test module | What it covers |
|---|---|
| `test_emd.py` | `emd`/`emd2` correctness vs POT; both solvers; n,m ≤ 200; tolerance 1e-6 on cost, 1e-9 on feasibility |
| `test_sparse_utils.py` | dense→CSR thresholding; scipy/torch sparse→CSR roundtrip; edge cases (all-zero, single edge, fully dense) |
| `test_routing.py` | solver selection vs threshold; `solver=` override; threshold table load |
| `test_lemon_accuracy.py` | float64 patch validation; sweep of cost magnitudes; relative error < 1e-6 across all cases |

**CI (GitHub Actions):** matrix over Python 3.10/3.11/3.12 on Linux and macOS. Runs on every PR.

---

## 8. PyPI Publishing

- **Package name:** `sparse-ot`
- **Build backend:** scikit-build-core (pybind11 extensions via CMakeLists.txt)
- **Wheel building:** `cibuildwheel` in CI; targets Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64, Python 3.10–3.12
- **Publishing:** GitHub Actions workflow triggered on `v*` tag push; uses PyPI Trusted Publisher (no stored API tokens)
- **Dependencies:** `numpy`, `scipy` (hard); `torch` as optional extra (`pip install sparse-ot[torch]`)
- **License:** MIT

---

## 9. POT Merge-Readiness

- Python module layout mirrors `ot/lp/` structure; `emd.py` can be integrated as `ot/lp/sparse_emd.py`
- API is a strict superset of POT's `emd`/`emd2` — no breaking changes
- Backend abstraction reuses POT's `Backend` protocol
- C++ layer is self-contained in `src/cpp/` with no Python coupling
- LEMON patch is documented and isolated in `src/cpp/lemon/PATCHES.md`
- Benchmark suite and routing threshold table are standalone and can be contributed separately
