# Sparse Bonneel + Solver Consolidation

**Date:** 2026-05-18
**Status:** Draft for review

## Goal

Make Bonneel's `NetworkSimplexSimple` the **only** OT solver in this package, and give it a sparse-input path whose memory and per-pivot cost scale with `k` (the number of candidate arcs) rather than `n·m`.

After this change:

- {Bonneel-dense, Bonneel-sparse} is the entire solver surface.
- LEMON `CostScaling` and OR-Tools are removed entirely.
- The public Python API matches POT's `ot.emd` / `ot.emd2` signatures.

## Motivation

- **Memory.** The current dense Bonneel path is O(n·m), which is the binding constraint on large problems.
- **Speed.** Bonneel's `NetworkSimplexSimple` is generally faster than LEMON `NetworkSimplex` due to flat-array layout, block-search pivot, and tight constants. By restoring those properties on top of a sparse digraph, we should keep that constant-factor advantage on sparse problems too.
- **Correctness.** LEMON `CostScaling` was found to return suboptimal flows on some inputs (recorded in the now-deleted suboptimality spec). Removing it eliminates that correctness risk.
- **Surface area.** OR-Tools was added as a fallback for LEMON's defects. With LEMON gone and Bonneel covering both dense and sparse, OR-Tools has no remaining role.

## Approach

Bonneel's `NetworkSimplexSimple<GR, V, C, A>` is templated on the digraph type `GR`. The O(n·m) memory comes entirely from instantiating it with `FullBipartiteDigraph`, which forces `arc_num = n·m`. The simplex itself only touches the graph via `nodeNum()`, `arcNum()`, `source(arc)`, `target(arc)`, and global arc/node iteration — it does not exploit density.

Instantiating the same `NetworkSimplexSimple` with a **sparse** bipartite digraph that exposes the same minimal LEMON Digraph concept gives, automatically:

- `_source`, `_target`, `_cost`, `_flow`, `_state` arrays of size O(k) instead of O(n·m).
- Block-search pivot cost O(k) per pivot instead of O(n·m).
- Same pivot rule, same flat-array layout, same constants.

No fork of `network_simplex_simple.h`; pure template re-instantiation.

## Components

### `src/cpp/bonneel/bipartite_sparse_digraph.h` (new)

A minimal LEMON-Digraph-concept type backed by caller-owned CSR arrays. Zero owned storage beyond two pointers and three ints.

Surface (only what `NetworkSimplexSimple::init()` and the pivot actually call):

```
typedef int     Node;
typedef int64_t Arc;          // matches Bonneel's ArcsType template arg

int     nodeNum()  const;     // n1 + n2
int64_t arcNum()   const;     // k = nnz
Node    source(Arc a) const;  // upper_bound(row_ptr, a) - 1
Node    target(Arc a) const;  // col_idx[a] + n1

void first(Arc&)  const;  void next(Arc&)  const;    // global arc iteration, a = k-1 .. 0
void first(Node&) const;  void next(Node&) const;    // global node iteration

static Arc  arcFromId(int64_t id);   // identity
static int  id(Node n);              // identity
static int64_t id(Arc a);            // identity
```

Deliberately **not** implemented (not called by the simplex on the paths we use): `firstOut`/`nextOut`/`firstIn`/`nextIn`, `arc(s,t)`, `findArc`.

`source(a)` uses binary search on `row_ptr` (O(log n) per call). It is only called during `init()`, once per arc, total O(k log n) — negligible vs. one simplex iteration. An explicit `row_of_arc[k]` array would make it O(1) but cost an extra O(k) ints; not worth it unless profiling says otherwise.

### `src/cpp/bonneel_solver.cpp` (extend both entry points)

**Both `solve_dense` and `solve_sparse` are modified to also return the dual potentials.** `NetworkSimplexSimple` already computes them; we expose `_pi[0..n-1]` as `u` (source duals) and `_pi[n..n+m-1]` as `v` (sink duals) via the existing `potential(node)` accessor (`network_simplex_simple.h:1033`).

```cpp
// new return type for solve_dense: (G, u, v)
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
solve_dense(a, b, M, numItermax);

// new return type for solve_sparse: (rows, cols, vals, u, v)
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse(a, b, row_ptr, col_idx, costs, numItermax);
```

Sign convention: Bonneel's reduced cost is `c_ij + pi[i] - pi[j]` (`network_simplex_simple.h:488`). To match POT's `u[i] + v[j] <= M[i,j]` convention, emit `u[i] = -pi[i]` and `v[j] = pi[n+j]`. (Verify against POT during implementation; flip signs if needed.)

Sparse-path body:

1. Build `BipartiteSparseDigraph di(n, m, row_ptr, col_idx)`.
2. `NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(di, true, n+m, k, numItermax);`
3. `net.supplyMap(a, n, neg_b, m);` — same negate-sink trick as `solve_dense`.
4. For `i in 0..k-1`: `net.setCost(di.arcFromId(i), costs[i]);`
5. `net.run();` — ignore the `INFEASIBLE` return; marginals are validated Python-side.
6. Build flow output: for `i in 0..k-1`, if `net.flow(arc i) > eps`, emit `(source(i), col_idx[i], flow(i))`.
7. Build duals: `u[i] = -net.potential(node i)` for `i in 0..n-1`; `v[j] = net.potential(node n+j)` for `j in 0..m-1`.

### `src/sparse_ot/emd.py` (simplify dramatically, plumb duals)

```python
def emd(a, b, M, numItermax=100000, log=False, center_dual=True):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / a.sum()
    b = b / b.sum()

    if scipy.sparse.issparse(M):
        row_ptr, col_idx, costs, n, m, _ = to_csr(M, 0.0)
        check_feasibility(a, b, row_ptr, col_idx)
        rows, cols, vals, u, v = _bonneel.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
        G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    else:
        M_dense = np.ascontiguousarray(M, dtype=np.float64)
        G, u, v = _bonneel.solve_dense(a, b, M_dense, numItermax)

    if center_dual:
        shift = u.mean()
        u = u - shift
        v = v + shift

    if log:
        cost = float((G.multiply(M).sum()) if scipy.sparse.issparse(G)
                     else np.sum(G * np.asarray(M)))
        return G, {"cost": cost, "u": u, "v": v,
                   "warning": None, "result_code": 1}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False):
    if log or return_matrix:
        G, info = emd(a, b, M, numItermax=numItermax, log=True)
        cost = info["cost"]
        if return_matrix:
            info = {**info, "G": G}
            return (cost, info) if log else (cost, G)
        return (cost, info) if log else cost
    G = emd(a, b, M, numItermax=numItermax, log=False)
    M_arr = M.toarray() if scipy.sparse.issparse(M) else np.asarray(M, dtype=np.float64)
    return float((G.multiply(M_arr)).sum()) if scipy.sparse.issparse(G) \
        else float(np.sum(G * M_arr))
```

POT-compatible behavior:

- `log=True` returns `(G, log_dict)` with keys `cost`, `u`, `v`, `warning`, `result_code` (matching POT's keys exactly).
- `center_dual=True` (POT default) applies the gauge shift `u -= u.mean(); v += u.mean()` so `u + v` is unchanged and `u` has zero mean. When `False`, raw simplex potentials are returned.
- No `solver=`, no `cost_sparsity_threshold=`, no `ortools_cost_scale=`.

### Removals

| Path | Action |
| --- | --- |
| `src/cpp/lemon_solver.cpp` | delete |
| `src/cpp/lemon/` (vendored tree) | delete |
| `src/sparse_ot/ortools_solver.py` | delete |
| `src/sparse_ot/routing.py` | delete (dispatch is now `issparse(M)`) |
| `src/sparse_ot/_ext/_lemon*` | delete (pybind module + build target) |
| `pyproject.toml` ortools / lemon extras and deps | delete |
| `benchmarks/bench_solvers.py` LEMON + OR-Tools columns and the densify-with-penalty workaround | delete |
| `tests/test_lemon_*`, `tests/test_ortools_*` | delete |

`feasibility.check_feasibility`, `sparse_utils.to_csr`: kept, still required by the sparse Bonneel path.

## Memory model after the change

Per-call memory:

- CSR inputs: 4(n+1) + 4k + 8k = O(k + n) bytes.
- Sparse digraph view: O(1).
- `NetworkSimplexSimple` internal arrays: `_source[k+n+m]`, `_target[k+n+m]`, `_cost[k+n+m]`, `_flow[k+n+m]`, `_state[k+n+m]` ints/doubles plus `_pi[n+m]` and a handful of `O(n+m)` tree-structure arrays. Total ≈ `O(k + n + m)`.

For a (10k × 10k, k = 100k) problem: ≈ a few MB, vs. ≈ 800 MB for the dense path on the same problem.

## Error handling

- **k = 0, n+m > 0** with nonzero `a` or `b`: caught by `check_feasibility` before we enter C++.
- **Disconnected support**: caught by `check_feasibility`.
- **`net.run()` returns `INFEASIBLE`**: proceed and return flows; Python validates marginals. Same accepted floating-point quirk as the dense path.
- **Cost dtype**: `double` throughout. No int64 scaling.

## Testing

- `tests/test_bonneel_sparse.py` — hand-crafted small problems (n ≤ 5), assert marginals exact, costs match `solve_dense` on the same problem.
- Property: random sparse problems with **full** support; sparse Bonneel cost must equal dense Bonneel cost within 1e-9.
- Property: random sparse problems with k-NN support; assert marginals, assert optimal cost ≤ dense cost on the same support (tautology, but catches regressions).
- Memory smoke test: (10k × 10k, k = 100k) problem; assert RSS peak < 200 MB via `resource.getrusage`.
- **Dual potentials:** for every test, assert `u[i] + v[j] <= M[i,j] + eps` for all i,j (feasibility) and `u[i] + v[j] == M[i,j]` within tolerance on every arc carrying positive flow (complementary slackness). Assert `cost == a @ u + b @ v` within tolerance (strong duality).
- **POT compat:** `tests/test_pot_compat.py` — for a small dense problem, call `ot.emd(..., log=True)` and our `emd(..., log=True)`; assert the same cost, marginals, and `u`, `v` matching up to a gauge constant (i.e. `u_ours - u_pot` is constant across i, and same for v with opposite sign). With `center_dual=True` the constant should be zero.
- Existing Bonneel-dense tests stay green; their assertions extend to also check the returned duals.

## Benchmarks

`benchmarks/bench_solvers.py` becomes a Bonneel-only comparison: dense vs. sparse on increasing problem sizes and densities. Drop LEMON and OR-Tools columns; drop the densify-with-penalty branch.

## Out of scope

- Adaptive / column-generation support refinement (the "Approach C" from brainstorming).
- Multi-threaded pivot rule.
- Routing/heuristic improvements to the upstream candidate-support selection.
- Any change to `feasibility.py` or `sparse_utils.py` beyond keeping them as-is.

## File-level change summary

**New:**
- `src/cpp/bonneel/bipartite_sparse_digraph.h`
- `tests/test_bonneel_sparse.py`

**Modified:**
- `src/cpp/bonneel_solver.cpp` — add `solve_sparse` pybind entry.
- `src/sparse_ot/emd.py` — simplify to two-branch dispatch on `issparse(M)`.
- `CMakeLists.txt` — drop LEMON target; sparse digraph header is header-only.
- `pyproject.toml` — drop ortools dep.
- `benchmarks/bench_solvers.py` — Bonneel-only.
- `src/sparse_ot/__init__.py` — drop solver-name exports if any.

**Deleted:**
- `src/cpp/lemon_solver.cpp`, `src/cpp/lemon/**`
- `src/sparse_ot/ortools_solver.py`
- `src/sparse_ot/routing.py`
- `tests/test_lemon_*`, `tests/test_ortools_*`
