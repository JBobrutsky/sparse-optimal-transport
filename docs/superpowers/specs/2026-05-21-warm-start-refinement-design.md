# Warm-start refinement for sparse-OT

**Date:** 2026-05-21
**Status:** Design — pending implementation
**Related work:** Schmitzer 2016 (sparse multiscale OT), Rauch & Zanotti 2025
(GridOT, arXiv:2502.20905), Bonneel 2011 (network simplex)

## Motivation

`sparse-ot` currently exposes one solve path: a cold `emd(a, b, M)` that calls
Bonneel's network simplex. Many real workflows produce a cheap candidate
solution first (e.g., a small-k nearest-neighbor approximation, or a previous
solve on a related problem) and would like to *refine* that candidate to a
globally optimal solution on a richer support without paying for a cold solve
from scratch.

The technique exists in the literature as Schmitzer's sparse multiscale
algorithm (the technique GridOT applies to grid-structured problems). This
spec adds it to `sparse-ot` in a **structure-agnostic** form: the caller
supplies the warm-start, the package does the verify-and-extend step.

## Non-goals

- No automatic restricted-support construction. Callers build their own
  warm-start.
- No grid-aware coarsening hierarchy. The technique is general; grid structure
  is the caller's prerogative.
- No callable-cost / lazy-materialization variant (Approach C from
  brainstorming). Possible future extension.
- No dense `M_full` support in v1. Refinement requires CSR `M_full`.

## API

Single change to the public surface: `emd` gains two kwargs.

```python
def emd(a, b, M, numItermax=None, log=False, center_dual=True,
        warm_start=None, reduced_cost_tol=None):
    ...
```

- `warm_start`: `None` (default, cold path), or the 2-tuple `(G, info)`
  returned by a prior `emd(..., log=True)` call. The flow `G` (CSR) is needed
  so we can return immediately when `(u, v)` is already dual-feasible on
  `E_full` — without it, we'd have to re-solve `M_full` from scratch just to
  produce a flow. `info` supplies the dual potentials `u`, `v`. A bare tuple
  `(G, u, v)` is also accepted for callers who didn't request `log=True`.
- `reduced_cost_tol`: `None` (default) picks `1e-9 * max(1, |M|_∞)`. Override
  in the rare case the cost scale is unusual.

When `warm_start` is not `None`:
- `M` must be CSR. Dense `M` with `warm_start` raises `NotImplementedError`.
- The returned `(G, info)` has the same shape as the cold path, plus an extra
  `info["refine"]` dict (see "Log dict additions" below).

Backward compatibility: `warm_start=None` is a byte-for-byte no-op on the
existing path.

### Example

```python
import numpy as np, scipy.sparse, sparse_ot as sot

# Phase 1 — cheap cold solve on a coarse support.
M_coarse = build_knn_cost(points, k=8)
G_coarse, info = sot.emd(a, b, M_coarse, log=True)

# Phase 2 — refine to optimum on a denser support. The full (G, info) tuple
# is the warm-start payload: G_coarse lets us return immediately when the
# warm-start is already optimal on M_full; info supplies (u, v).
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

## Architecture

```
emd(a, b, M, warm_start=None, ...)
        │
        ▼
   warm_start is None?
        │           │
       yes          no
        │           │
        ▼           ▼
  existing       refine_from_warm_start(...)
  cold path      (new module: src/sparse_ot/refine.py)
  (unchanged)         │
                      ▼
               One vectorized verifier pass over nnz(M_full),
               then at most one Bonneel-sparse re-solve on M_full
               warm-started from (u, v).
```

**Invariants:**
1. `warm_start=None` → byte-identical to today.
2. Result is provably optimal on `(a, b, M_full)` (LP duality).
3. The `info` returned is the same shape as `info` consumed — chainable.
4. Memory: O(nnz(M_full) + nnz(warm-start support)) for CSR input.
5. CSR-only for the new path in v1.

## Algorithm

Given `(u, v)` from a prior solve on `S_warm ⊆ E_full`:

1. **Validate** `warm_start` shape and finiteness (see "Error handling").
2. **Compute reduced costs** on every edge of `M_full`:
   ```
   row_idx = np.repeat(np.arange(n), np.diff(M_full.indptr))   # O(nnz), once
   rc      = M_full.data - u[row_idx] - v[M_full.indices]      # vectorized
   ```
3. **Decide**:
   - If `min(rc) ≥ -reduced_cost_tol`: `(u, v)` is dual-feasible on `E_full`,
     and `G_warm` is primal-feasible for `(a, b)` (by assumption — it came
     from a valid prior solve) with support in
     `S_warm ⊆ {(i, j) : rc[i, j] ≈ 0}`. By complementary slackness `G_warm`
     extended with zeros over `E_full \ S_warm` is a global optimum on
     `(a, b, M_full)`. **Return immediately** — no re-solve needed.
   - Else: violating edges define entering variables. We re-solve
     Bonneel-sparse on the full `M_full` support, warm-started from `(u, v)`
     when the C++ binding supports it (see "Pybind / C++ extension").
4. **Return** `(G, info)` matching the cold-path contract, with `info["refine"]`
   populated.

**Correctness:** by LP duality, a primal feasible flow whose support consists
of edges with `rc = 0` and whose duals are `(u, v)` is optimal. The
network-simplex re-solve enforces primal feasibility; the verifier confirms
(or the re-solve restores) dual feasibility on `E_full`.

**v1 simplification — "single-pass" refinement.** Because we pass the full
`M_csr` to the verifier and the re-solve, the path is always one of two
straight lines: the warm-start is already dual-feasible on `E_full` (return
`G_warm` directly, 0 solves), or it isn't and we re-solve once on the full
support (1 solve). We keep the design loop-shaped for the future
callable-cost extension, but the v1 implementation has no loop.

## Components

### New module: `src/sparse_ot/refine.py`

Single public entry:
```python
def refine_from_warm_start(a, b, M_csr, warm_start, *,
                           numItermax, reduced_cost_tol, center_dual):
    ...
```

Helpers (private):
- `_parse_warm_start(warm_start, n, m) -> (G, u, v)`: accepts the `(G, info)`
  tuple, the bare `(G, u, v)` tuple, validates shapes/finiteness/CSR-ness of
  `G`, raises with field-level messages.
- `_compute_reduced_costs(M_csr, u, v) -> (rc, min_rc, n_violating)`.
- `_resolve_with_warm_start(a, b, M_csr, u, v, numItermax) -> (G, u', v')`:
  thin wrapper over `_bonneel.solve_sparse` that passes the warm-start duals
  (extension to the pybind layer described below).

### Router change in `emd.py`

```python
if scipy.sparse.issparse(M):
    if warm_start is not None:
        return refine_from_warm_start(a, b, M, warm_start, ...)
    # existing cold sparse path
elif warm_start is not None:
    raise NotImplementedError(
        "warm_start is only supported for sparse (CSR) M in v1"
    )
```

### Pybind / C++ extension

Bonneel's network simplex supports initializing the basis from supplied dual
potentials (and re-deriving an initial basic feasible solution). We expose a
new entry point:

```cpp
// returns (rows, cols, vals, u, v) same shape as solve_sparse
auto solve_sparse_warm(a, b, row_ptr, col_idx, costs, numItermax, u0, v0);
```

If exposing this is non-trivial in the upstream Bonneel implementation, v1
falls back to **cold re-solve on `M_full`** in the non-optimal branch (still
produces the optimum; only loses pivot-count savings on that re-solve). The
"warm-start was already optimal" branch is independent of the C++ change —
that branch never re-solves; it returns `G_warm` re-shaped onto `M_full`'s
support directly. So the warm-start-was-already-optimal case is essentially
free *because* we receive `G_warm`, not because of any C++ change.

## Data flow

```
Inputs: a, b (marginals), M_csr (full CSR), warm_start (dict or tuple)

  ├─ _parse_warm_start ── (G_warm, u, v)
  ├─ check_feasibility(a, b, M_csr)  ── existing check on full support
  ├─ _compute_reduced_costs(M_csr, u, v) ── (rc, min_rc, n_violating)
  │
  ├─ if min_rc >= -tol:
  │     # Reshape G_warm onto M_csr's index layout — same values, larger
  │     # CSR with zeros on the new edges. No solve.
  │     G = _reshape_onto_support(G_warm, M_csr.shape, M_csr)
  │     refine_info = {warm_start_optimal: True, num_passes: 0, ...}
  │
  └─ else:
        G, u, v = _resolve_with_warm_start(...)   # full re-solve
        refine_info = {warm_start_optimal: False, num_passes: 1, ...}

  Post:
    center_dual shift (existing)
    _check_marginals (existing)
    pack info dict (existing keys + "refine")
```

## Log dict additions

```python
info["refine"] = {
    "warm_start_optimal": bool,        # was (u, v) dual-feasible on E_full?
    "num_passes": int,                 # re-solves performed (0 or 1 in v1)
    "initial_min_reduced_cost": float, # min(rc) before any refinement
    "edges_added": int,                # |S_final| - |S_warm|; 0 when already optimal
}
```

Cold path does **not** populate `info["refine"]`. Presence of the key
distinguishes refinement output from cold output.

## Error handling

| Condition | Behavior |
|---|---|
| `warm_start` missing `G`, `u`, or `v` | `TypeError` with field-level message |
| `len(u) != n` or `len(v) != m` or `G.shape != (n,m)` | `ValueError` |
| `u`/`v` contains non-finite values | `ValueError` (warm-start corrupted) |
| `G_warm` support not a subset of `M_full` support | `ValueError` (warm-start incompatible with `M_full`) |
| `warm_start` provided with dense `M_full` | `NotImplementedError` |
| `M_full` empty (nnz == 0) | `InfeasibleProblemError` via existing check |
| `a`/`b` shape mismatch with `M_full` | existing `ValueError` (unchanged) |

**Feasibility precondition:** we still call `check_feasibility(a, b, ...)` on
`M_full` — the warm-start being feasible on a subset gives no guarantee about
`M_full`'s support.

**Numerical edge cases:**
- `reduced_cost_tol` defaults to `1e-9 * max(1, |M|_∞)`, scale-relative.
- We do not verify `a`, `b` match what the warm-start was solved against.
  Misuse manifests as a high `initial_min_reduced_cost` and an extra solve,
  not as wrong output.
- Reduced costs are shift-invariant under `(u, v) → (u+c, v-c)`, so
  `center_dual=True`/`False` warm-starts are both accepted.

**No silent fallback** to cold path when the warm-start is malformed (we
raise). A merely *non-optimal* warm-start is the normal case and triggers the
re-solve.

**Post-solve marginal check:** identical to the cold path — `RuntimeWarning`
+ `result_code=0` if max marginal violation exceeds `1e-6`.

## Testing strategy

Test file: `tests/test_refine.py` (new). Existing tests untouched.

### Correctness

1. **Already-optimal warm-start round-trip.** Cold solve → feed `(G, info)`
   back as `warm_start` on the same `(a, b, M)` → assert
   `warm_start_optimal=True`, `num_passes=0`, costs and plans identical.
2. **Sub-support → full-support convergence.** Cold solve on `M_restricted`
   (k=5 NN), refine on `M_full` (k=20 NN); assert cost matches a fresh cold
   solve on `M_full` to within `1e-9`.
3. **Regime sweep.** Parametrize `(n, k_warm, k_full)` over the seeded
   `knn-grid` problems; assert refined cost equals cold cost.
4. **Wrong-marginals warm-start.** Use `(u, v)` solved against `(a, b)` to
   warm-start a refinement on `(a', b')`. Still returns the correct optimum
   on `(a', b')`.
5. **Centered vs uncentered duals.** Same result both ways — shift-invariance.

### Errors

6. Mismatched `u` length → `ValueError`.
7. NaN in `v` → `ValueError`.
8. Dense `M_full` with `warm_start` → `NotImplementedError`.
9. `warm_start=({}, {})` (missing keys) → `TypeError`.
10. Bare tuple form `warm_start=(G, u, v)` produces the same result as the
    `(G, info)` form.
11. `G_warm` with an edge outside `M_full`'s support → `ValueError`.

### Benchmark (separate file)

`benchmarks/bench_refine.py`. Sweep `n ∈ {1k, 10k, 100k}`,
`k_full ∈ {32, 128, 512}`, `k_warm/k_full ∈ {0.1, 0.25, 0.5, 1.0}`. Three
timings per cell: `cold_full`, `refine` (combined), `refine_amortized`
(refinement only). Each cell asserts cost equality between cold and refined
paths within `1e-6` relative.

## File layout

| Path | Change |
|---|---|
| `src/sparse_ot/refine.py` | New |
| `src/sparse_ot/emd.py` | Add `warm_start`, `reduced_cost_tol` kwargs |
| `src/sparse_ot/__init__.py` | No new exports |
| `src/cpp/bonneel_solver.cpp` | Add `solve_sparse_warm` binding (optional v1) |
| `tests/test_refine.py` | New |
| `benchmarks/bench_refine.py` | New |
| `benchmarks/results/refine.json` | New output |
| `README.md` | Add "Warm-starting from a previous solve" section |
| `docs/refinement.md` | New long-form doc |

## Documentation deliverables

1. **Module docstring** in `refine.py` — canonical algorithm reference, with
   citations.
2. **README section** — the example above, plus a one-paragraph "when to use
   this" pointer to `docs/refinement.md`.
3. **`docs/refinement.md`** — long-form: algorithm in full, regime of
   optimality with measured numbers from the benchmark, "when NOT to use"
   subsection, and a worked **32×32** dual-feasibility example printing
   intermediate `(u, v, rc)` and the support before/after refinement
   (small enough to print, large enough to exhibit a non-trivial extension).
4. **Spec (this document)** — committed to
   `docs/superpowers/specs/2026-05-21-warm-start-refinement-design.md`.

## References

- Schmitzer, B. "A sparse multiscale algorithm for dense optimal transport."
  *Journal of Mathematical Imaging and Vision*, 56(2):238–259, 2016.
  https://doi.org/10.1007/s10851-016-0653-9
- Rauch, J. and Zanotti, L. "An improved implementation of Schmitzer's sparse
  multiscale algorithm for discrete optimal transport on grids."
  arXiv:2502.20905, 2025. https://arxiv.org/abs/2502.20905. Reference
  implementation: https://github.com/johannesrauch/GridOT (Boost license).
- Bonneel, N. et al. "Displacement interpolation using Lagrangian mass
  transport." *ACM TOG*, 30(6), 2011.
  https://github.com/nbonneel/network_simplex
- Bertsimas, D. and Tsitsiklis, J. *Introduction to Linear Optimization*,
  Athena Scientific, 1997. §4 column generation / dual feasibility test.

## Open questions for implementation

- Does Bonneel's upstream API expose enough to warm-start the basis from
  `(u, v)` cleanly? If not, v1 falls back to cold re-solve on `M_full` and
  only the "warm-start was already optimal" case is free. (The verifier pass
  is independent and still saves the work in that case.)
- What's the right `GROWTH_CAP` if/when we add the callable-cost variant?
  Out of scope for v1, noted for follow-up.
