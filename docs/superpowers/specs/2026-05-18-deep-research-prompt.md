# Deep Research Prompt: Efficient Exact Sparse Optimal Transport at Large Scale

Paste the section below into Claude chat with **Deep Research** enabled. The prompt is self-contained — Deep Research has no access to this repo.

---

## Prompt for Deep Research

I'm building a Python package (`sparse-ot`) that solves **exact** discrete optimal transport (no entropic regularization) on **very large, very sparse** bipartite cost graphs. The reference use case is restricted OT between two fully dense 3D volumes of shape 256³ ≈ 16.7M points each, where each source voxel is only allowed to transport to its ≤32 nearest neighbors in the target grid — sparsity ratio ≈ 2 × 10⁻⁶, total nonzeros ≈ 536M.

I have benchmarked three off-the-shelf solver paths and all three fail at the scale that matters:

### What I have tried, and where each breaks

1. **Bonneel's network-simplex** (the `network_simplex_simple.h` from N. Bonneel; the same code POT wraps as `ot.emd`). Solves on the dense n×n cost matrix.
   - **Wins** decisively up to n ≈ 10⁴ both in time (~7 s at n=10⁴, k=100) and accuracy (matches reference to 10 digits).
   - **Breaks** at n ≥ 30 000 because materializing the dense matrix is O(n²) memory; at n = 16.7M the dense matrix would be 280 PB.
   - It also cannot natively distinguish "absent edge" from "real edge with cost 0" on a sparse input — a separate footgun I have patched.

2. **LEMON's `CostScaling<int64 supplies, double costs>`** (Goldberg–Tarjan push/relabel). Floating-point-patched termination `_epsilon < 1e-9 × _initial_max_cost`, where `_initial_max_cost = N × max_cost`.
   - **Works** at n ≤ 10³ with usable accuracy.
   - **Suboptimal** at n = 1000, k = n (dense): 0.1–0.4% relative cost gap vs. POT/Bonneel. The math: per-edge precision floor is `1e-9 × max_cost`. For cost = (i−j)² on n=1000, max_cost ≈ 10⁶, so floor ≈ 10⁻³ — exactly matches the observed gap. Tightening `_tolerance` to 10⁻¹² closes the gap but blows runtime by ~8× per decade of tolerance.
   - **Hangs > 15 min** at n=100 000, k=32 (3.2M edges, max_cost≈10³) — the exact "≥100K nodes, sparse" regime LEMON's documentation promises CostScaling outperforms network simplex. Either the relative-tolerance termination is too loose to converge cleanly at this scale, or there is a deeper algorithmic / wrapper issue.

3. **Google OR-Tools `SimpleMinCostFlow`** (cost-scaling internally; int64 costs only, scaled from float64 by ×1e6).
   - **Works** but slow: at n=100K, k=32 it takes 33 min and peaks at **76 GB** memory — orders of magnitude more than the theoretical edge-list footprint. Unusable beyond ~1M edges on commodity hardware.
   - Achieves only `feasibility_a ≈ 3e-7` marginal residual at that scale.

### My empirical observations from running these solvers on a 16 GB Mac

| n | k | Bonneel | LEMON | OR-Tools |
|---|---|---|---|---|
| 1 000 | 1 000 (dense) | 0.07 s, optimal | 20 s, 0.1% gap | 27 s, ~10⁻⁵ |
| 10 000 | 100 (sparse) | 7 s, 1.9 GB | hung > 30 s | 144 s, 0.76 GB |
| 100 000 | 32 (sparse) | OOM (80 GB dense) | hung > 15 min | 33 min, 76 GB |
| 16 700 000 | 32 (target use case) | OOM | unknown — never reached | OOM |

The reference 3D-voxel problem has n = 16.7M and ≈ 32 outgoing edges per source. Total active edges ≈ 5 × 10⁸. The cost values are squared Euclidean distances ≤ 32 (small dynamic range, helpful for any scaling algorithm). Marginals are dense full-support distributions (every voxel has positive mass).

### What I need from this research

Produce a structured survey + implementation recommendation answering, in order:

1. **What algorithms exist for exact min-cost flow / discrete OT that scale to n ≈ 10⁷, nnz ≈ 5 × 10⁸ on a single workstation or a small cluster, without entropic regularization?** Include:
   - Modern variants of network simplex tailored for sparse graphs (e.g., LEMON's `NetworkSimplex`, COIN-OR's MCFClass, hybrids that don't materialize dense matrices).
   - Improved cost-scaling implementations (e.g., Goldberg's CS2, Bertsekas–Tseng RELAX, BIM, LEMON ISO variants). Note their termination criteria and how they handle floating-point costs.
   - Auction-type algorithms (Bertsekas 1988+) and ε-scaling variants. These are well-studied for very-sparse assignment-like problems; are there implementations that handle non-square / non-unit-supply OT?
   - Out-of-core / streaming approaches if any.
   - GPU-parallel exact MCF — Hopcroft–Karp variants, parallel auction, push-relabel on GPU. Are any of these production-quality?
   - 1-D / structured-grid OT shortcuts: for cost = squared distance on a regular grid with k-NN support, are there exact closed-form or near-linear-time algorithms (e.g., via sorted-marginals 1-D reduction, or via grid-graph specific shortest-path tricks)?

2. **For each candidate algorithm, what is its empirical and theoretical behavior on the regime I care about: n ≈ 10⁵–10⁷, k = O(1) per source, cost dynamic range ≤ 10²?** Cite measured runtimes from papers, library benchmarks, or independent comparisons. Highlight: does the algorithm depend on `max_cost` for its termination, like LEMON's CostScaling does? If so, what's the asymptotic precision floor?

3. **Why does LEMON CostScaling hang on my n=100K, k=32 problem when its documentation explicitly recommends it for "≥100K nodes, especially sparse"?** Is this a known issue with the float64 patch? Is there a different LEMON algorithm (NetworkSimplex, CapacityScaling) that handles this regime better? Look at LEMON's mailing list, issue tracker, and the COIN-OR cost-scaling papers.

4. **Has anyone solved exact OT at the n ≈ 10⁷, sparse-k scale specifically for image registration / restricted OT on 3D volumes?** Domain: medical imaging, video frame interpolation, point-cloud registration. Cite papers and their solver choices. Is the consensus "use entropic Sinkhorn and accept the regularization bias" or has someone actually pushed exact OT to that scale?

5. **Concrete implementation recommendation.** Given:
   - The package is already pybind11 + C++ with vendored solvers,
   - We can vendor more C++ (LEMON-style) or call external libraries via subprocess,
   - The target user runs on a 32–64 GB Mac/Linux workstation, GPU optional,
   - We want to ship an MIT/BSD-compatible solver,
   
   what is the single best algorithm + implementation to target *next* for the n ≈ 10⁵ regime, and what for the n ≈ 10⁷ reference case? Spell out the path: which library or which paper to implement, what its expected runtime and memory will be, and what the precision contract should look like.

6. **Routing implications.** Currently I pick a solver based on `(n, nnz)` thresholds derived empirically. Given the survey above, what's the right *axis* to route on? Just (n, k)? Or also `max_cost / min_cost`? Or the bipartite graph's structural properties (regular grid, k-NN, arbitrary)?

### Output format

- Executive summary (≤ 200 words)
- Algorithm landscape table (algorithm × scale × accuracy × memory × license × maturity)
- Drill-down on the 2–3 most promising paths with citations
- A specific "next step" recommendation for my n ≈ 10⁵ regime, and a separate one for n ≈ 10⁷
- Open questions / things the literature doesn't answer

Cite papers (with arXiv / DOI where available), library URLs (GitHub), and benchmark sources. Prefer post-2015 work but include foundational papers (Goldberg–Tarjan 1990, Bertsekas auction, etc.) where the algorithmic content is still current.
