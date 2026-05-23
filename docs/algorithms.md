# Technical Reference: sparse-ot

This document covers the algorithms, design decisions, and academic context behind `sparse-ot`. For the API and usage examples, see the [README](../README.md).

---

## Background

Discrete optimal transport (OT) seeks a transport plan $T \in \mathbb{R}^{n \times m}_{\geq 0}$ minimising the total cost

$$\min_{T \geq 0} \sum_{i,j} T_{ij} M_{ij} \quad \text{s.t.} \quad T\mathbf{1} = a,\; T^\top\mathbf{1} = b$$

where $a \in \mathbb{R}^n$ and $b \in \mathbb{R}^m$ are probability vectors and $M$ is a cost matrix. This is a linear programme over a bipartite flow network, solvable exactly in polynomial time via network simplex algorithms.

The dominant bottleneck in practice is not the solver itself but the cost matrix: materialising all $n \times m$ entries requires $O(nm)$ memory and $O(nm)$ work per pivot. For $n = m = 10{,}000$ this exceeds 800 MB before a single pivot runs. Many real problems — k-NN graphs, point-cloud matching, grid-structured transport, restricted transport on image volumes — are *intrinsically sparse*: only $k \ll nm$ pairs are meaningful candidate transport edges.

`sparse-ot` eliminates the $O(nm)$ barrier by exploiting sparsity throughout: in memory layout, in the pivot cost, and in the warm-start refinement.

---

## Contribution 1: Sparse instantiation of Bonneel's network simplex

**Key reference:** Bonneel et al., *Displacement interpolation using Lagrangian mass transport*, ACM Transactions on Graphics 30(6), 2011. ([code](https://github.com/nbonneel/network_simplex))

Bonneel's `NetworkSimplexSimple` is a high-performance network simplex solver templated on a digraph type `GR`. The [POT library](https://github.com/PythonOT/POT) instantiates it with `FullBipartiteDigraph`, which forces all internal arrays (`_source`, `_target`, `_cost`, `_flow`, `_state`) to size $O(nm)$ and makes each block-search pivot scan all $nm$ arcs.

`sparse-ot` provides a new digraph type, `BipartiteSparseDigraph`, backed by caller-owned CSR arrays. It satisfies the minimal LEMON Digraph concept required by `NetworkSimplexSimple` without owning any data beyond two pointers. Instantiating the same solver template over this new digraph type yields automatically:

- Internal arrays of size $O(k)$ where $k = \text{nnz}(M)$.
- Block-search pivot cost $O(k)$ instead of $O(nm)$.
- Identical pivot rule, flat-array layout, and numerical constants as the dense path.

This is a *template re-instantiation*, not a fork: `network_simplex_simple.h` is unmodified.

**Memory comparison** for a $10{,}000 \times 10{,}000$ problem with $k = 100{,}000$ candidate edges:

| Path | Memory | Per-pivot work |
|---|---|---|
| Bonneel-dense (POT default) | ≈ 800 MB | $O(nm) = O(10^8)$ |
| **Bonneel-sparse (sparse-ot)** | **≈ 6 MB** | $O(k) = O(10^5)$ |

The dense path is infeasible on this instance on a typical laptop; the sparse path runs in seconds.

### Feasibility checking

The sparse path restricts transport to the edges in $M$. Before invoking the solver, `sparse-ot` runs a connected-components pass over the bipartite support graph (via `scipy.sparse.csgraph.connected_components`) in $O(k)$ time. For each connected component, it verifies that the supply from source nodes equals the demand at target nodes within tolerance. If any component is imbalanced, `InfeasibleProblemError` is raised with the violating component's indices and mass imbalance before any solver call is made.

Hall's condition is a necessary and sufficient condition for a feasible flow when all edge capacities are unbounded. The component-balance check is equivalent to Hall's condition in this setting.

### Dual potentials and POT compatibility

Both dense and sparse paths expose the dual variables $(u, v)$ from the network simplex (`_pi` array). With `center_dual=True` (default), $u$ is shifted to zero mean so that $u[i] + v[j]$ is unchanged on every arc — matching POT's sign convention ($u[i] + v[j] \leq M_{ij}$ at optimum, with equality on arcs in the optimal basis). The log dict keys (`cost`, `u`, `v`, `warning`, `result_code`) match POT's exactly.

---

## Contribution 2: Structure-agnostic warm-start refinement

**Key references:**
- Schmitzer, B., *A sparse multiscale algorithm for dense optimal transport*, Journal of Mathematical Imaging and Vision 56(2):238–259, 2016. [doi:10.1007/s10851-016-0653-9](https://doi.org/10.1007/s10851-016-0653-9)
- Rauch, J. and Zanotti, L., *An improved implementation of Schmitzer's sparse multiscale algorithm for discrete optimal transport on grids*, arXiv:2502.20905, 2025. [arXiv](https://arxiv.org/abs/2502.20905) | [GridOT](https://github.com/johannesrauch/GridOT)

Schmitzer (2016) showed that a solution on a restricted support $S_\text{warm} \subset E_\text{full}$ can be extended to a globally optimal solution on $E_\text{full}$ cheaply, by checking dual feasibility and selectively adding violated edges. GridOT (Rauch & Zanotti 2025) applies this to grid-structured problems.

`sparse-ot` implements this technique in a **structure-agnostic** form: the caller supplies the warm start; the package performs the verify-and-extend step without assuming any grid or hierarchy.

### Algorithm

Given a prior solve $(G_\text{warm}, u, v)$ on support $S_\text{warm}$ and a full support $M_\text{full}$ (CSR):

1. **Reduced cost computation** — vectorised over all $\text{nnz}(M_\text{full})$ edges:
   $$r_{ij} = M_{ij} - u_i - v_j$$

2. **Decision** — if $\min_{(i,j) \in E_\text{full}} r_{ij} \geq -\tau$ (where $\tau = 10^{-9} \max(1, \|M\|_\infty)$ by default): the duals $(u, v)$ are feasible on $E_\text{full}$. By LP complementary slackness, $G_\text{warm}$ extended with zeros on $E_\text{full} \setminus S_\text{warm}$ is a global optimum. **Return immediately — no re-solve.**

3. **Fallback** — otherwise, re-solve on $M_\text{full}$ warm-started from $(u, v)$. The returned plan is provably optimal on $(a, b, M_\text{full})$ by LP duality.

**Correctness guarantee:** the returned plan is always a global optimum on $(a, b, M_\text{full})$, regardless of which branch fires.

### When warm-starting wins

The "return immediately" branch (step 2) fires whenever the warm-start duals are already dual-feasible on $E_\text{full}$. This happens when:
- Same problem, expanded support (e.g., $k\text{-NN}$ with $k$ increased): if the optimum is realised inside $S_\text{warm}$, the extended solution is already globally optimal.
- Same support, perturbed metric: nearby cost functions often share near-identical dual variables.

Measured speedups at $n = 16{,}000$:
- **Support expansion** (`warm_ratio = 0.95`, $k_\text{warm} \approx k_\text{full}$): 140–450× over a cold solve on $M_\text{full}$.
- **Metric change** (L2² → L1, same support): ~23× over a cold L1 solve.

When the warm-start duals are not feasible on $E_\text{full}$, the fallback re-solve is required; see Contribution 3.

### Chaining

The `info` dict returned by `emd(..., log=True)` is accepted as the `warm_start` payload directly, so refinement chains are written naturally:

```python
G1, info1 = sot.emd(a, b, M_coarse, log=True)
G2, info2 = sot.emd(a, b, M_medium, warm_start=(G1, info1), log=True)
G3, info3 = sot.emd(a, b, M_full,   warm_start=(G2, info2), log=True)
```

---

## Contribution 3: Full-basis warm start for network simplex

**Background:** Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Athena Scientific, 1997. §4 (basis warm-starting and column generation).

When Contribution 2's fallback re-solve fires, a cold re-solve would discard the prior spanning tree and restart from the artificial star, wasting pivots re-discovering what the prior solve already established. This is suboptimal when the warm-start duals are close but not dual-feasible (e.g., same support with a perturbed metric).

The full-basis warm start replaces the cold re-solve with a spanning tree injection into `NetworkSimplexSimple`:

1. Extract a spanning tree of size $n + m - 1$ from $G_\text{warm}$ using union-find, preferring high-flow arcs.
2. Reconstruct the thread-list arrays (`_thread`, `_parent`, `_pred`, `_forward`, etc.) required by the pivot loop via DFS from the artificial root.
3. Inject $\pi$ from $(u_0, v_0)$ directly into `_pi`.
4. Set `_state`: warm arcs → `STATE_TREE`, remainder → `STATE_LOWER`.
5. Run the pivot loop — no artificial arcs need to be driven out when $G_\text{warm}$ is non-degenerate.

Two dispatch modes based on the degeneracy of $G_\text{warm}$:
- **Mode B (full basis):** used when $G_\text{warm}.\text{nnz} \geq (1 - \delta)(n + m - 1)$ for $\delta = 0.05$. Zero artificial overhead when non-degenerate.
- **Mode C (cold fallback):** used when degeneracy is high or the warm plan violates marginals. Falls back to a standard cold solve — injecting `_pi` into the artificial-star initialisation breaks dual consistency when the warm arcs are not in the basis, so no potential override is applied.

This change affects only the non-optimal fallback branch and is invisible to the caller. The already-optimal branch (Contribution 2, step 2) is independent and unchanged.

---

## Benchmark methodology

### Problem generation

All benchmark instances use the **marginals-from-plan** construction to guarantee feasibility:

1. Build the $k$-NN band graph on a 1D grid ($n = m$).
2. Sample edge weights $w_{ij} = \exp(\eta_{ij})$, $\eta_{ij} \sim \mathcal{N}(0, 1)$.
3. Set $a_i = \sum_j w_{ij}$ and $b_j = \sum_i w_{ij}$, then normalise both by $W = \sum_{ij} w_{ij}$.

By construction $w/W$ is a feasible transport plan from $a$ to $b$, so Hall's condition is satisfied for any $k \geq 1$. Independent Dirichlet draws routinely violate Hall's condition on narrow supports and are not used.

### Correctness

Accuracy is measured relative to `min(cost)` across all solvers that returned a primal-feasible plan on the same instance. When $k = n$ (fully dense), POT participates and provides cross-validation against an independent implementation. On sparse instances the reference is the agreement floor between `sparse-ot` and OR-Tools.

OR-Tools rounds costs to integers (scale factor $10^6$), so agreement with `sparse-ot` is bounded at $\sim 10^{-6}$ relative cost error; this is an OR-Tools precision limit, not a `sparse-ot` defect.

### Extrapolation

For solvers that reach a memory or time cutoff before large $n$, wall times are extrapolated via a power-law fit in log-log space:

$$\log t = \log a + b \log n + c \log k$$

using `scipy.optimize.curve_fit`. Extrapolated values are flagged `"extrapolated": true` in the JSON and rendered as dashed lines in figures. No extrapolation is shown if $R^2 < 0.95$.

### Benchmark cutoffs

| Constant | Value | Effect |
|---|---|---|
| `POT_MAX_N` | 2 000 | POT skipped when $n > 2{,}000$ |
| `POT_MAX_NNZ` | 100 000 | POT skipped when sparse nnz $> 10^5$ |
| `ORTOOLS_MAX_N` | 2 000 | OR-Tools skipped when $n > 2{,}000$ |
| `ORTOOLS_MAX_NNZ` | 500 000 | OR-Tools skipped when nnz $> 5 \times 10^5$ |

Skipped cells are recorded as `null` in the JSON so coverage gaps are visible. Raise constants in `benchmarks/solvers.py` for larger hardware.

---

## Convergence and the `numItermax` knob

Bonneel's network simplex stops at `numItermax` pivots without raising. The default is problem-size-aware:

$$\text{numItermax} = \min(5 \times 10^7,\ \max(10^5,\ 100 \cdot (n + m + k)))$$

After every solve, marginal violations are checked. If $\max(\|G\mathbf{1} - a\|_\infty, \|G^\top\mathbf{1} - b\|_\infty) > 10^{-6}$, a `RuntimeWarning` is emitted and `result_code = 0` is set in the log dict. No exception is raised, matching POT's behavior.

---

## References

1. Bonneel, N., van de Panne, M., Paris, S., and Heidrich, W. Displacement interpolation using Lagrangian mass transport. *ACM Transactions on Graphics*, 30(6), 2011. https://github.com/nbonneel/network_simplex

2. Schmitzer, B. A sparse multiscale algorithm for dense optimal transport. *Journal of Mathematical Imaging and Vision*, 56(2):238–259, 2016. https://doi.org/10.1007/s10851-016-0653-9

3. Rauch, J. and Zanotti, L. An improved implementation of Schmitzer's sparse multiscale algorithm for discrete optimal transport on grids. *arXiv:2502.20905*, 2025. https://arxiv.org/abs/2502.20905

4. Flamary, R. et al. POT: Python Optimal Transport. *Journal of Machine Learning Research*, 22(78):1–8, 2021. https://github.com/PythonOT/POT

5. Bertsimas, D. and Tsitsiklis, J. *Introduction to Linear Optimization*. Athena Scientific, 1997.
