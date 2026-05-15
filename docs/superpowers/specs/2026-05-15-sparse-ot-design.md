# sparse-ot Design Spec

**Date:** 2026-05-15  
**Status:** Approved

## Overview

`sparse-ot` is a Python package published to PyPI that provides drop-in replacements for POT's `emd()` and `emd2()` functions, optimized for heavily sparse bipartite graphs. It supports three solvers — Bonneel's network simplex (dense/near-dense cost matrices), float64-patched LEMON CostScaling (sparse, moderate scale), and OR-Tools min-cost flow (very large sparse problems) — and routes between them automatically based on empirically derived thresholds. The package is structured to facilitate future merge into the official POT project.

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
│   │   ├── ortools_solver.py        # OR-Tools min-cost flow (Python API, optional dep)
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
            cost_sparsity_threshold=0.0, solver=None, ortools_cost_scale=1e6)

# Transport cost — returns scalar
cost = sot.emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
                cost_sparsity_threshold=0.0, solver=None, ortools_cost_scale=1e6)
```

**Extensions beyond POT:**
- `cost_sparsity_threshold` (float, default 0.0): when `M` is a dense array, values with `|M[i,j]| <= cost_sparsity_threshold` are treated as absent edges. Exact zeros are always treated as absent regardless of this parameter.
- `solver` (str or None): `'bonneel'`, `'lemon'`, `'ortools'`, or `None` (auto-route). Bypasses routing when set.
- `ortools_cost_scale` (float, default 1e6): OR-Tools requires int64 costs; float64 costs are multiplied by this factor and rounded before being passed to OR-Tools, then the result is unscaled. Only relevant when `solver='ortools'` or OR-Tools is selected by the router.

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
    ├─ solver
    │   Bonneel (C++ ext): (a, b, M_dense float64) → dense transport matrix float64
    │   LEMON   (C++ ext): (a, b, row_ptr, col_idx, costs float64) → COO (i, j, val) float64
    │   OR-Tools (Python):  costs scaled to int64 → SimpleMinCostFlow → COO unscaled to float64
    │
    └─ backend.py
        wrap result in output format matching input backend
        LEMON / OR-Tools result assembled as scipy CSR (or torch sparse COO)
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

### OR-Tools solver (`ortools_solver.py`)

OR-Tools is consumed via its official Python package (`pip install sparse-ot[ortools]`), not vendored. No C++ wrapper is needed — OR-Tools' `SimpleMinCostFlow` Python API is used directly.

Since OR-Tools requires int64 costs and int64 node supplies, float64 inputs are scaled before solving and unscaled after:

```python
# ortools_solver.py
from ortools.graph.python import min_cost_flow

def solve_ortools(a, b, row_ptr, col_idx, costs, ortools_cost_scale=1e6):
    # Scale distributions to int64 supplies/demands
    # Scale costs to int64
    # Build SimpleMinCostFlow graph arc by arc
    # Solve and extract flow values
    # Unscale transport plan back to float64
    # Return COO (row_indices, col_indices, values)
```

OR-Tools is imported lazily inside `solve_ortools` so that the package remains importable without OR-Tools installed. A missing OR-Tools installation raises `ImportError` with a clear install message only when the OR-Tools solver is actually invoked.

---

## 5. Routing Logic

```python
# routing.py
def select_solver(n: int, m: int, nnz: int, solver: str | None = None) -> str:
    if solver is not None:
        return solver
    k = nnz / n   # average neighbors per source node
    thresholds = _load_thresholds(n, m)   # from routing_thresholds.json
    if k > thresholds['bonneel_lemon']:
        return 'bonneel'
    if n > thresholds['lemon_ortools']:
        return 'ortools'
    return 'lemon'
```

Routing is a two-threshold decision over `k` (neighbors per node) and `n` (problem size):
- **k above `bonneel_lemon` threshold** → Bonneel (cost matrix is near-dense, NS wins)
- **k below threshold AND n above `lemon_ortools` threshold** → OR-Tools (graph too large for LEMON)
- **otherwise** → LEMON

Both thresholds are stored in `benchmarks/results/routing_thresholds.json` and derived empirically from the benchmark sweep. Using `k` rather than raw sparsity ratio keeps the threshold numerically stable across very large n.

**Before benchmarks are run**, the package ships with conservative defaults: `bonneel_lemon = 128`, `lemon_ortools = 1_000_000`.

**The threshold table is regenerated** by running `python benchmarks/generate_report.py` and committing the updated JSON. The README documents this process for users who want hardware-specific routing.

---

## 6. Benchmark Suite

**Problem generator:** both distributions `a` and `b` are always fully dense (all n bins have positive mass, drawn from a Dirichlet distribution). Sparsity is controlled exclusively by `k` — the number of allowed neighbors per source node — so `nnz = k × n`. Problems are structured to match the reference use case: a regular grid in 1D/2D/3D where each source node connects to its `k` nearest neighbors in the target grid.

### Memory cutoffs

Benchmark cutoffs are defined as named constants in `bench_solvers.py` and documented in the README. Default values target a 16GB RAM machine:

| Constant | Default | Condition skipped |
|---|---|---|
| `MAX_DENSE_N` | 8 192 | Dense matrix `n×n×8 bytes > ~512MB`; Bonneel and POT reference skipped above this |
| `MAX_SPARSE_NNZ` | 200 000 000 | Sparse CSR `nnz×20 bytes > ~4GB`; LEMON skipped above this (OR-Tools only) |
| `MAX_ORTOOLS_NNZ` | 500 000 000 | OR-Tools memory limit; OR-Tools skipped above this |

Cells outside these limits are recorded as `null` in the results JSON (not skipped silently) so the coverage gap is visible in the report. The README notes that the reference case (n=16.7M, nnz=536M) exceeds `MAX_ORTOOLS_NNZ` on a 16GB machine and documents how to raise the limits on larger hardware.

### Efficiency sweep

```
n        ∈ {1K, 4K, 8K, 16K, 64K, 256K, 1M, 4M, 16M}
k        ∈ {2, 8, 32, 128, 512, 2048, n/10, n}     # neighbors per node; k=n is the fully dense case
solvers  = ['bonneel', 'lemon', 'ortools', 'pot_reference']
metrics  = wall_time (median of 5 runs), peak_memory_mb, iterations
```

Each `(n, k, solver)` cell is skipped (recorded as `null`) if `n > MAX_DENSE_N` for dense solvers, `k×n > MAX_SPARSE_NNZ` for LEMON, or `k×n > MAX_ORTOOLS_NNZ` for OR-Tools. Results: `benchmarks/results/efficiency.json`

### Accuracy sweep

For each `(n, k)` configuration where LEMON is the selected solver:
- **n ≤ 64K:** ground truth via POT's `emd2`; report relative cost error `|cost_lemon - cost_pot| / cost_pot` and primal feasibility `‖T @ 1 − a‖∞`, `‖Tᵀ @ 1 − b‖∞`
- **n > 64K:** POT cannot run at this scale; report primal feasibility only (no cost error metric)

Results: `benchmarks/results/accuracy.json`

### Report figures (publication-quality, PDF + PNG)

- **Heatmap:** wall time ratio over the (n, k) grid for each solver pair (LEMON/Bonneel, OR-Tools/LEMON); crossover contours mark the two routing thresholds
- **Line plots:** wall time vs. n at fixed k values, one line per solver
- **Accuracy plot:** relative cost error and feasibility residual vs. k per solver, with confidence bands across random problem instances
- **Routing threshold derivation:** annotated crossover contours used to generate `routing_thresholds.json`

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
- **Dependencies:** `numpy`, `scipy` (hard); `torch` as optional extra (`pip install sparse-ot[torch]`); `ortools` as optional extra (`pip install sparse-ot[ortools]`); both together via `pip install sparse-ot[torch,ortools]`
- **License:** MIT

---

## 9. POT Merge-Readiness

- Python module layout mirrors `ot/lp/` structure; `emd.py` can be integrated as `ot/lp/sparse_emd.py`
- API is a strict superset of POT's `emd`/`emd2` — no breaking changes
- Backend abstraction reuses POT's `Backend` protocol
- C++ layer is self-contained in `src/cpp/` with no Python coupling
- LEMON patch is documented and isolated in `src/cpp/lemon/PATCHES.md`
- Benchmark suite and routing threshold table are standalone and can be contributed separately
