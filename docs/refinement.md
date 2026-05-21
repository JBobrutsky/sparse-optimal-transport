# Warm-start refinement

A long-form companion to `src/sparse_ot/refine.py`. Read the module
docstring for the algorithm and citations; this document covers regime
of optimality and a worked numerical example.

## When to use this path

Use `warm_start` when:

* you already have an OT solution `(G, info)` from a previous `emd` call;
* you want a globally optimal solution on a **larger** sparse support
  containing the previous one (e.g., the same problem at a higher
  nearest-neighbor `k`);
* `M_full` is sparse (CSR) — dense `M_full` is not supported in v1
  because it defeats the memory benefit.

Do **not** use this path when:

* `M_full` is dense — call cold `emd(a, b, M)` instead.
* you have no prior solve — same answer, cold is fine.
* the prior support is unrelated to the optimum on `M_full` — the
  verifier costs `O(nnz(M_full))` and you'll still pay for a cold
  re-solve.

## Regime of optimality

Empirically, on the seeded `knn-grid` problems (see
`benchmarks/bench_refine.py`):

* When `S_warm = E_full` (the warm-start ran on the *same* support as
  `M_full`): refinement is essentially the verifier pass — sub-millisecond
  per `nnz`. Speedup is the entire cold-solve cost.
* When `S_warm` is a strict subset that captures the optimum (e.g., the
  same problem on a tighter `k`): refinement matches cold-on-`M_full` in
  cost; wall time is dominated by Phase 1 (the user's cheap solve) plus a
  cheap verify pass. The "warm_start_optimal=True" branch fires.
* When `S_warm` is a strict subset that *misses* the optimum:
  refinement falls back to a cold re-solve on `M_full` (v1 limitation —
  see below). The verifier pass becomes overhead. Wall time is
  Phase 1 + cold-on-`M_full`. Use cold `emd(a, b, M_full)` directly in
  this regime.

The exact crossover depends on Bonneel's pivot-count sensitivity to the
warm-start basis, which v1 does not yet exploit (cold re-solve fallback —
see the design spec, "Pybind / C++ extension").

## Worked example — 32×32 dual-feasibility check

A 32×32 problem is small enough to print and large enough to actually
exhibit a non-trivial refinement.

```python
import numpy as np, scipy.sparse, sparse_ot as sot

n = 32
rng = np.random.default_rng(0)

def band_cost(n, k):
    half = k // 2
    rows, cols, costs = [], [], []
    for i in range(n):
        lo = max(0, i - half); hi = min(n, lo + k); lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i); cols.append(j); costs.append(float((i - j) ** 2))
    return scipy.sparse.csr_matrix(
        (costs, (rows, cols)), shape=(n, n)
    )

# Feasible marginals on the k=3 band (sums to 1 by construction).
M3 = band_cost(n, k=3)
w = np.exp(rng.standard_normal(M3.nnz)); w /= w.sum()
a = np.zeros(n); b = np.zeros(n)
np.add.at(a, M3.nonzero()[0], w); np.add.at(b, M3.nonzero()[1], w)
a /= a.sum(); b /= b.sum()

# Phase 1 — solve on the tight k=3 band.
G3, info3 = sot.emd(a, b, M3, log=True)
print("Phase 1 cost on k=3:", info3["cost"])
print("Phase 1 nnz(G):", G3.nnz)

# Phase 2 — refine on a wider k=9 band. The same problem on the larger
# support has the same optimal cost iff the optimum is realised inside the
# k=3 band, which it is by construction (marginals were sampled from k=3).
M9 = band_cost(n, k=9)
G9, info9 = sot.emd(a, b, M9, warm_start=(G3, info3), log=True)
print("Phase 2 refine info:", info9["refine"])
# {'warm_start_optimal': True, 'num_passes': 0,
#  'initial_min_reduced_cost': ≈0.0, 'edges_added': 0}
print("Phase 2 cost on k=9:", info9["cost"])
assert abs(info9["cost"] - info3["cost"]) < 1e-12

# Force a non-trivial refinement: tilt the marginals so the optimum on k=9
# uses edges outside the k=3 support.
a2 = np.roll(a, 1); b2 = np.roll(b, -1)
a2 /= a2.sum(); b2 /= b2.sum()
# Re-solve k=3 (still feasible — band-3 is connected on n=32).
G3_t, info3_t = sot.emd(a2, b2, M3, log=True)
G9_t, info9_t = sot.emd(a2, b2, M9, warm_start=(G3_t, info3_t), log=True)
print("Tilted refine info:", info9_t["refine"])
# warm_start_optimal=False; num_passes=1; edges_added>0.
print("Tilted Phase 2 cost on k=9:", info9_t["cost"])
G9_cold, info9_cold = sot.emd(a2, b2, M9, log=True)
assert abs(info9_t["cost"] - info9_cold["cost"]) < 1e-9
```

Reading the output:

* `initial_min_reduced_cost` is the minimum of
  `M_full[i, j] − u[i] − v[j]` over every nnz of `M_full`. A value ≥ 0
  (within tolerance) means the warm-start's duals are already feasible on
  the larger support, and refinement returns `G_warm` directly. Due to
  floating-point arithmetic, "zero" is reported as a small negative number
  in the range `[−tol, 0]` even when the warm-start is optimal; `tol`
  defaults to `1e-9 * max(1, ‖M‖_∞)`.
* `num_passes` is 0 when the already-optimal branch fires (no re-solve)
  and 1 when the cold re-solve fallback is used.
* `edges_added` is the change in `nnz(G)` from `G_warm` to `G_refined`. In
  the already-optimal branch it is 0 by construction; in the cold-re-solve
  branch it is the difference in the solvers' chosen bases.
* `warm_start_optimal` is the bit that distinguishes the two branches.
