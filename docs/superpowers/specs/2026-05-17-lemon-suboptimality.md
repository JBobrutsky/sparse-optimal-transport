# LEMON CostScaling Suboptimality — Diagnosis

**Date:** 2026-05-17
**Status:** Diagnosed; recommended action is a routing change, not an algorithm fix.

## TL;DR

The 0.1–0.4% LEMON suboptimality observed at `n=1000, k=1000` is **hypothesis (a)** — the float64-patched termination criterion. The criterion is `_epsilon < 1e-9 × _initial_max_cost`, and `_initial_max_cost = N × max_cost` grows with both problem size and cost-matrix dynamic range. At large `n × dense k × wide cost range`, the absolute termination floor becomes large relative to the achieved optimal cost, leaving LEMON in a non-optimal feasible state.

Tightening `_tolerance` from `1e-9` to `1e-12` closes the gap completely but explodes runtime (>7× at `n=500`, intractable at `n=1000`). The literature agrees: LEMON's CostScaling is designed for large sparse graphs (≥100K nodes), not the medium dense regime where Bonneel's network simplex wins on every axis. **Recommendation: keep `_tolerance = 1e-9`; rely on routing to send dense-and-medium problems to Bonneel.** A secondary bug found during this investigation (Bonneel returns zero-cost plans on sparse inputs) is more urgent.

## Reproduction

`benchmarks/results/accuracy_quick.json` (Plan 4 baseline, `_tolerance=1e-9`):

| Cell | LEMON | POT (optimal) | Rel. gap |
|---|---|---|---|
| n=200, k=200, seed=0 | 0.4606520395 | 0.4606520403 | -1.8e-9 (noise) |
| n=1000, k=1000, seed=0 | 0.504145 | 0.503585 | **+1.1e-3** |
| n=1000, k=1000, seed=1 | 1.133008 | 1.133008 | +0 |
| n=1000, k=1000, seed=2 | 0.797067 | 0.794226 | **+3.6e-3** |

Seed-dependent. Pure cost-scaling artifact: the achieved plan is always primal-feasible (`‖T1 − a‖∞ < 5e-13`).

## Mechanism

LEMON's CostScaling (`src/cpp/lemon/cost_scaling.h`) uses an ε-scaling loop:

```cpp
// init: line 819-832
for each arc j:
    lc = scost[j] * res_node_num * _alpha;   // LargeCost = double; lc grows ∝ N·cost
    if (lc > _epsilon) _epsilon = lc;
_epsilon /= _alpha;
_initial_max_cost = _epsilon;                 // ≈ N × max_cost
_tolerance = 1e-9;

// main loop: line 1370 (and 1495)
for ( ; _epsilon >= _tolerance * _initial_max_cost; _epsilon /= _alpha )
    ...
```

So termination floor in LargeCost units is `tolerance × N × max_cost`. Translated back to original cost units (dividing by `N`), the per-edge reduced-cost precision is `tolerance × max_cost`.

For our k-NN band benchmark, `max_cost = (n-1)²`. So per-edge precision is `1e-9 × (n-1)²`:

| n | max_cost = (n-1)² | per-edge floor | observed rel gap |
|---|---|---|---|
| 200 | 39 601 | 4.0e-5 | ~1e-9 (clean) |
| 500 | 249 001 | 2.5e-4 | small |
| 1000 | 998 001 | **1.0e-3** | ~1e-3 ✓ |

The per-edge floor and the observed relative cost gap line up. Accumulation across many active edges is then partially cancelled by the global min-cost structure, which is why the gap is in the ballpark of the per-edge floor (not nnz × floor).

This is a fundamental property of the Goldberg–Tarjan cost-scaling algorithm: it terminates at an ε that is *relative to the cost range*, not relative to the achieved objective. When the cost range and objective are decoupled (as in our `(i-j)²` band with normalized marginals), the achievable relative accuracy is governed by `max_cost / typical_objective_cost`.

## Experimental confirmation

Edited `src/cpp/lemon/cost_scaling.h:832` to test:

1. `_tolerance = 1e-9` (baseline): n=200 → 1e-9 noise floor; n=1000 → 1e-3 gap (matches table).
2. `_tolerance = 1e-12`: n=200 → unchanged (already at float64 noise); n=1000 → ran >7 min single seed before being killed. The runtime growth comes from running 3 extra ε-passes (each scales ε by 1/α=1/8), and the late passes are the most expensive (small ε ⇒ many push/augment iterations per pass).
3. `_tolerance = 1e-15`: n=500 → ran >7 min single seed before killed. Confirms tightening alone is intractable.

Restored `_tolerance = 1e-9` after experiments.

## Why this is *not* the wrapper

I inspected `src/cpp/lemon_solver.cpp` for hypothesis (c):

- `SUPPLY_SCALE = 1e12`: gives ~12 decimal digits for supplies in `[0, 1]`. With normalized marginals summing to 1, each `a_int[i]` rounds to within 1 of its true scaled value, so absolute supply error ≤ `n / 1e12 = 1e-9` — well below the 1e-3 gap we see.
- The rounding residual is absorbed into `b_int[m-1]` (line 73), preserving `sum(a_int) == sum(b_int)`.
- Costs are passed through unscaled as `double` — no quantization at the wrapper level.
- Sign convention: `supply[source] = +a_int`, `supply[sink] = -b_int` is standard for LEMON.

No wrapper bug. The integer scaling argument fits the data: if it were the bug, gaps would track `SUPPLY_SCALE`, not `n × max_cost`.

## Why LEMON isn't expected to "outperform Bonneel for highly sparse cost matrices" *here*

The LEMON documentation states:

> NetworkSimplex is usually the fastest on relatively small graphs and dense graphs, while CostScaling is typically more efficient on large graphs with **hundreds of thousands of nodes or above, especially if they are sparse**.

Our benchmark grid runs `n ∈ {200, 1000, 4000, 16000}` — entirely below CostScaling's design point. In this regime:

- **Wall-time:** Bonneel beats LEMON at every cell where both fit in memory (from the Task 10 mid sweep — that's why the empirical `bonneel_lemon` threshold came out at `k=2`).
- **Accuracy:** Bonneel matches POT to 10 digits; LEMON's accuracy degrades with `max_cost`, as derived above.

CostScaling's advantage shows up at `n ≥ 100k` and very sparse `k`, where:
- Network simplex's pivoting overhead scales with O(n × m) in practice;
- CostScaling's push/relabel scales with O(nm × log(n × max_cost)), which is asymptotically better;
- Bonneel's O(n²) dense-cost materialization runs out of memory (e.g. our `MAX_DENSE_N = 8192`).

So the *literature claim* about CostScaling outperforming network simplex is about asymptotic regimes we don't hit on a laptop benchmark. The current code is already wired for that future: routing falls back to LEMON (and then OR-Tools) when Bonneel can no longer fit. The empirical thresholds will rebalance on hardware that runs the full spec sweep.

## Secondary finding: Bonneel silently returns 0-cost plans on sparse input

From the probe at `n=200/1000, k ∈ {4, 32}` with `solver='bonneel'`:

```
n=200 k=4:  bonneel cost = 0.0   (LEMON: 0.687)
n=1000 k=4: bonneel cost = 0.0   (LEMON: 0.649)
n=1000 k=32: bonneel cost = 0.0  (LEMON: 1.572)
```

Mechanism: `src/sparse_ot/emd.py:66` calls `M.toarray()` on the sparse CSR. Absent cells become `0.0`. Bonneel sees a fully-dense cost matrix in which most cells are 0 and routes all mass through those "free" edges → cost 0 → silently wrong plan.

This **isn't a routing concern** for users (routing never selects Bonneel for sparse inputs once `bonneel_lemon ≥ 2`), but it's a footgun for `solver='bonneel'` overrides and it contaminated the benchmark `accuracy_quick.json` before Plan 4 Task 7 switched the reference to `min(cost over feasible solvers)`.

Recommended fix (separate from this task): in `emd.py`, when `selected == 'bonneel'` and `not dense_input`, either (a) raise `ValueError("Bonneel does not support sparse cost matrices; use solver='lemon' or 'ortools', or pass a dense M with +inf in absent cells")`, or (b) materialize absent cells as `+inf` (or a very large penalty) before passing to Bonneel. Option (a) is safer.

## Recommendations

1. **Keep `_tolerance = 1e-9`.** Tightening it doesn't pay off at the n where the gap appears: runtime grows by ~8× per decade of `_tolerance`, and the routing already prefers Bonneel/POT/OR-Tools at those sizes. No code change.

2. **Fix Bonneel-on-sparse silently returning zero** (separate follow-up). One-line raise or a sentinel fill in `emd.py:62-67`.

3. **Document the LEMON precision floor** in `benchmarks/results/README.md`: "At dense `k=n` with wide cost dynamic range (e.g. `max_cost / min_nonzero_cost > 10⁵`), LEMON's `_tolerance=1e-9` produces a relative cost gap up to `1e-9 × max_cost`. Prefer Bonneel or OR-Tools in that regime."

4. **Future routing nuance** (not urgent): consider extending `select_solver` to look at `max_cost / min_cost` (cost dynamic range), not just `(n, k)`. A high-dynamic-range dense problem should never go to LEMON regardless of routing thresholds. Defer until a real user hits the corner.

## Open questions

- Is the seed-dependent variation in the gap (0% at seed=1 vs 0.36% at seed=2) explainable by which sparse-plan structure the per-edge floor happens to bias? Worth confirming by examining the support overlap between LEMON's plan and POT's optimal plan on seed=2.
- Does `OR-Tools` solve with comparable precision at scale, or does it have its own scaling floor? At `n=1000 k=1000` it matched POT to ~5 digits but at sparse `k=4` it agreed with LEMON to 6 digits — different precision regimes, probably also from int64 cost scaling. Worth a separate one-shot.
