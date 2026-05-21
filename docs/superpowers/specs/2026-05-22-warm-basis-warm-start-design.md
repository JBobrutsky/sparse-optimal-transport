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

### Degeneracy and non-basic G_warm

`G_warm.nnz` relative to `n+m-1` (the size of a valid spanning tree) determines mode selection:

**`G_warm.nnz < n+m-1` (degenerate — typical solver output)**
Zero-flow basic arcs are dropped by the solver (`if (f > eps)`), so the gap `n_degenerate = (n+m-1) - G_warm.nnz` counts lost tree arcs. When the gap is large, spanning tree reconstruction is approximate and Mode B degrades toward Mode C.

**`G_warm.nnz > n+m-1` (non-basic — user-constructed or perturbed)**
G_warm is not a basic feasible solution (e.g., spatially perturbed optimal plan, convex combination of solutions). `n_degenerate` is negative, so the threshold check does not fire — Mode B is used. Before C++ dispatch, `G_warm.eliminate_zeros()` is called first (stored zeros inflate `nnz`); if `nnz` still exceeds `n+m-1`, a `RuntimeWarning` is emitted but Mode B proceeds. The union-find in C++ handles cycles gracefully by skipping arc-forming arcs — it extracts a valid spanning tree of exactly `n+m-1` arcs and the remainder become `STATE_LOWER`.

Speedup in the non-basic case is primarily driven by **dual quality** (how close `(u, v)` are to dual-feasible on `M_full`), not spanning tree structure. To maximise spanning tree quality when G_warm is non-basic, arcs are **sorted by flow descending** before union-find: high-flow arcs are more likely to be in the optimal basis and are therefore preferred as tree arcs. Sorting costs O(nnz log nnz) in Python before the C++ call.

**Mode selection summary:**

| `G_warm.nnz` vs `n+m-1` | Action |
|---|---|
| Equal (non-degenerate) | Mode B, no sort needed |
| Slightly less (mild degeneracy) | Mode B, C++ fills gaps with artificial arcs |
| Much less (> 5% gap) | Warn + Mode C |
| Greater (non-basic) | `eliminate_zeros()`, sort by flow desc, Mode B with warning if still > n+m-1 |

**O(1) threshold check:** after `eliminate_zeros()`, compute `n_degenerate = (n+m-1) - G_warm.nnz`. If `n_degenerate / (n+m-1) > _DEGENERATE_WARN_THRESHOLD` (default `0.05`), emit `RuntimeWarning` and use Mode C.

---

## Architecture

```
emd(a, b, M_full, warm_start=(G, info))
  └─ refine_from_warm_start()             [refine.py — Python]
       ├─ already-optimal branch          [unchanged]
       └─ non-optimal branch
            └─ bonneel_sparse_solve_warm() [sparse_utils.py — Python]
                 ├─ eliminate_zeros()      fix inflated nnz
                 ├─ _degenerate_check()    O(1): (n+m-1) - G_warm.nnz
                 ├─ sort by flow desc      O(nnz log nnz), improves union-find quality
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

    # Drop stored zeros before measuring nnz (inflated by scipy CSR bookkeeping).
    G_warm = G_warm.copy()
    G_warm.eliminate_zeros()

    n_degenerate = n_basic - G_warm.nnz

    if n_degenerate < 0:
        # G_warm is non-basic (e.g. user-constructed or perturbed plan).
        warnings.warn(
            f"warm_start G has {-n_degenerate} more nonzeros than a spanning tree "
            f"(nnz={G_warm.nnz}, n+m-1={n_basic}). G_warm is not a basic feasible "
            "solution; arcs will be sorted by flow and a spanning tree extracted.",
            RuntimeWarning, stacklevel=4,
        )
        n_degenerate = 0  # treat as non-degenerate; sort handles quality

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
        # Sort by flow descending so union-find in C++ prefers high-flow arcs
        # as tree arcs — they are more likely to be in the optimal basis.
        order = np.argsort(coo.data)[::-1]
        rows, cols, vals, u, v = _bonneel.solve_sparse_warm_basis(
            a, b, row_ptr, col_idx, costs, u0, v0,
            coo.row[order].astype(np.int32),
            coo.col[order].astype(np.int32),
            coo.data[order].astype(np.float64),
            numItermax,
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
| `test_warm_basis_non_basic_G_warn` | `G_warm.nnz > n+m-1` after `eliminate_zeros` → `RuntimeWarning`, correct result |
| `test_warm_basis_non_basic_sort_helps` | Perturbed near-optimal G (non-basic): cost matches cold, `warm_basis_used=True` |

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
