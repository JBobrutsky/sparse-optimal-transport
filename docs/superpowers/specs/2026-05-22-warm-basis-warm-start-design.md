# Design: Full-Basis Warm Start for Network Simplex

**Date:** 2026-05-22
**Branch target:** `warm-start-refinement` (extends existing warm-start API)
**Status:** Approved for implementation

---

## Problem

The v1 warm-start refinement (`refine_from_warm_start`) handles two cases:

- `warm_start_optimal=True`: duals `(u, v)` are feasible on `M_full` → return `G_warm` directly, zero solver calls.
- `warm_start_optimal=False`: duals are infeasible → **cold re-solve** on `M_full` from scratch.

The cold re-solve discards everything from the prior solve. When the warm-start duals are close to optimal on `M_full` (e.g., same support with a perturbed metric, or a strict subset support that captures most of the optimum), the simplex wastes pivots re-discovering what the prior solve already knew.

---

## Goal

Replace the cold re-solve fallback with a **full-basis warm start**: inject the prior spanning tree and dual potentials into the network simplex before the first pivot, so the solver starts from a near-optimal basis rather than an artificial star.

This preserves the existing public API (`emd`, `warm_start` kwarg, info dict contract) and the already-optimal branch entirely. Only the non-optimal branch changes.

---

## Approach

Two warm-start modes, dispatched by Python based on degeneracy:

**Mode B — Full basis injection (primary)**
Extract a spanning tree from `G_warm`, build the network simplex tree metadata (parent/pred/thread arrays) in C++, inject `_pi` from `(u0, v0)`, and run. Artificial arcs are used only for nodes not covered by `G_warm` (degenerate gap arcs). When `G_warm` is non-degenerate, zero artificial arcs enter the pivot loop.

**Mode C — Potential-only (fallback)**
Standard artificial-star initialization, then override `_pi` from `(u0, v0)`. Used when `G_warm` is severely degenerate (many zero-flow basic arcs dropped by the prior solve). Fewer real-arc pivots than cold, but artificial arc overhead remains.

### Why full basis beats potential-only

A non-degenerate network simplex solution has exactly `n+m-1` basic arcs forming a spanning tree. Potential-only keeps the artificial star — even with perfect potentials, the solver must drive all `n+m` artificial arcs out of the basis before it can terminate. Full basis pre-loads the real spanning tree so no artificial arc overhead exists.

### Degeneracy

Zero-flow basic arcs are dropped from `G` by the solver (`if (f > eps)`). So `G_warm.nnz ≤ n+m-1`; the gap is the number of degenerate arcs lost. When the gap is large, spanning tree reconstruction from `G_warm` is approximate and the benefit of Mode B over Mode C diminishes.

**O(1) early check:** `n_degenerate = (n+m-1) - G_warm.nnz`. If `n_degenerate / (n+m-1) > _DEGENERATE_WARN_THRESHOLD` (default `0.05`), emit a `RuntimeWarning` and use Mode C instead.

---

## Architecture

```
emd(a, b, M_full, warm_start=(G, info))
  └─ refine_from_warm_start()             [refine.py — Python]
       ├─ already-optimal branch          [unchanged]
       └─ non-optimal branch
            └─ bonneel_sparse_solve_warm() [sparse_utils.py — Python]
                 ├─ _degenerate_check()    O(1): (n+m-1) - G_warm.nnz
                 ├─ Mode B path
                 │    └─ _bonneel.solve_sparse_warm_basis()   [C++]
                 │         warmBasisInit(): union-find + DFS + _pi inject
                 └─ Mode C path
                      └─ _bonneel.solve_sparse_warm_potentials() [C++]
                           overridePotentials(): _pi inject only
```

**Unchanged:** `emd()` signature, info dict keys, `solve_sparse` (cold path + Phase 1), `_parse_warm_start`, `_verify_support_subset`, `_compute_reduced_costs`, already-optimal branch.

---

## C++ Layer

### New entry points in `bonneel_solver.cpp`

**`solve_sparse_warm_basis(a, b, row_ptr, col_idx, costs, u0, v0, warm_rows, warm_cols, warm_flows, numItermax)`**

Builds `BipartiteSparseDigraph` + `NetworkSimplexSimple` identically to `solve_sparse`, then calls `net.warmBasisInit(u0, v0, warm_rows, warm_cols, warm_flows, n_warm)` in place of the standard artificial-star block inside `run()`. Returns the same `(rows, cols, vals, u, v)` tuple.

**`solve_sparse_warm_potentials(a, b, row_ptr, col_idx, costs, u0, v0, numItermax)`**

Standard initialization (artificial star via `run()`), then calls `net.overridePotentials(u0, v0)` to overwrite `_pi` before the first pivot. Returns the same tuple.

### New methods on `NetworkSimplexSimple`

**`warmBasisInit(u0, v0, warm_src, warm_tgt, warm_flow, n_warm)`**

Replaces the artificial-star initialization block in `run()`. Steps:

1. Set `_pi[i] = -u0[i]` for source nodes `i ∈ [0, n)`, `_pi[n+j] = v0[j]` for target nodes `j ∈ [0, m)`. Sign convention matches the negation applied in `solve_sparse` when extracting duals (`up[i] = -net.potential(di(i))`).
2. Union-find over the `n_warm` warm arcs to build a spanning forest of the `n+m` real nodes.
3. DFS from the artificial root (`_root = n+m`) to construct thread-list arrays: `_thread`, `_rev_thread`, `_succ_num`, `_last_succ`, `_parent`, `_pred`, `_forward`. These are required by the pivot loop's subtree-update routines.
4. Attach any unspanned nodes (degenerate gap) to `_root` via artificial arcs at `ART_COST`, exactly as the cold init does — degenerate cases degrade gracefully to a mixed warm/artificial basis.
5. Set `_state`: warm arcs → `STATE_TREE`, all others → `STATE_LOWER`.

`warmBasisInit` is called from a new `runWarm()` method that mirrors `run()` but substitutes the artificial-star block with `warmBasisInit`. This avoids any conditional flag inside `run()` itself — the existing cold path is untouched.

**`overridePotentials(u0, v0, n, m)`**

Writes `_pi[i] = -u0[i]` / `_pi[n+j] = v0[j]` after the artificial-star initialization. ~5 lines.

---

## Python Layer

### `sparse_utils.py` — new helper

```python
_DEGENERATE_WARN_THRESHOLD = 0.05  # fraction of n+m-1

def bonneel_sparse_solve_warm(a, b, row_ptr, col_idx, costs, n, m,
                               G_warm, u0, v0, numItermax=None):
    """Warm-started network simplex: full basis (Mode B) or potential-only (Mode C)."""
    if numItermax is None:
        numItermax = _default_num_iter(n, m, len(costs))
    n_basic = n + m - 1
    n_degenerate = n_basic - G_warm.nnz
    if n_degenerate > _DEGENERATE_WARN_THRESHOLD * n_basic:
        warnings.warn(
            f"warm_start G has {n_degenerate} zero-flow basic arcs "
            f"({100 * n_degenerate / n_basic:.0f}% of the spanning tree). "
            "Basis reconstruction will be approximate; using potential-only warm start.",
            RuntimeWarning, stacklevel=4,
        )
        warm_basis_used = False
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_potentials(
            a, b, row_ptr, col_idx, costs, u0, v0, numItermax
        )
    else:
        warm_basis_used = True
        coo = G_warm.tocoo()
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_basis(
            a, b, row_ptr, col_idx, costs, u0, v0,
            coo.row.astype(np.int32), coo.col.astype(np.int32),
            coo.data.astype(np.float64), numItermax,
        )
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    return G, u, v, warm_basis_used
```

### `refine.py` — non-optimal branch

Replace:
```python
G, u, v = bonneel_sparse_solve(a, b, row_ptr, col_idx, costs, n, m, numItermax)
```

With:
```python
G, u, v, warm_basis_used = bonneel_sparse_solve_warm(
    a, b, row_ptr, col_idx, costs, n, m, G_warm, u, v, numItermax
)
```

### `refine.py` — updated `refine_info`

```python
refine_info = {
    "warm_start_optimal": False,
    "num_passes": 1,
    "initial_min_reduced_cost": min_rc,
    "edges_added": max(G.nnz - G_warm.nnz, 0),
    "warm_basis_used": warm_basis_used,   # True=full basis, False=potential-only
}
```

---

## Testing

### `tests/test_warm_basis.py` (new file)

| Test | What it checks |
|---|---|
| `test_warm_basis_exact_same_support` | Same support + same metric: cost matches cold, `edges_added=0` |
| `test_warm_basis_same_support_different_metric` | Same support + perturbed costs: cost matches cold solve on perturbed metric |
| `test_warm_potentials_fallback_correctness` | Mode C path: correct result when forced via high degeneracy |
| `test_warm_basis_sign_convention` | 2×2 analytic problem: injected `_pi` matches cold-solve potentials |

### `tests/test_refine.py` additions

| Test | What it checks |
|---|---|
| `test_warm_basis_degenerate_warn` | `G_warm.nnz << n+m-1` → `RuntimeWarning` emitted, result still correct |
| `test_warm_basis_threshold_fires_potential_only` | Same + assert `refine_info["warm_basis_used"] == False` |
| `test_warm_basis_subset_support` | M_full ⊃ G_warm support: non-optimal branch takes warm basis path |
| `test_warm_basis_different_metric_close` | Phase 1 on L1, Phase 2 on L2: cost matches cold L2 solve |

### `benchmarks/bench_refine.py` additions

Add `warm_basis_sec` and `warm_basis_amortized_sec` columns to each benchmark cell. The existing correctness gate (`rel < 1e-6`) covers the new path without changes.

---

## Build

C++ changes require recompilation. Implementation plan must include a build step (`pip install -e .`) before any test run.

---

## Out of scope

- Exposing the spanning tree in the `info` dict (Approach A) — deferred; would require changing the solver's return contract.
- Warm-starting the dense path (`solve_dense`) — not needed; dense M is already rejected by the warm-start API.
- Warm-starting across different `(n, m)` shapes — the existing `_parse_warm_start` shape guard rejects this before warm-start dispatch.
