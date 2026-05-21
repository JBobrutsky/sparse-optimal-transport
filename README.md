# sparse-ot

[![CI](https://github.com/JBobrutsky/sparse-optimal-transport/actions/workflows/ci.yml/badge.svg)](https://github.com/JBobrutsky/sparse-optimal-transport/actions/workflows/ci.yml)

Drop-in replacement for [POT](https://github.com/PythonOT/POT)'s `emd` /
`emd2`, with native support for **sparse cost matrices**. One solver
(Bonneel's network simplex) covers both regimes:

- **Dense** `numpy.ndarray` cost matrix → dense plan.
- **`scipy.sparse` CSR** cost matrix → sparse plan, with memory and per-pivot
  work both scaling in the number of candidate edges `k` rather than `n × m`.

## Quickstart

```python
import numpy as np, scipy.sparse, sparse_ot as sot

# Dense: identical interface to ot.emd.
G = sot.emd(a, b, M)
cost = sot.emd2(a, b, M)

# Sparse: pass a CSR cost matrix. Absent entries are forbidden edges, not
# zero-cost shortcuts.
M_csr = scipy.sparse.csr_matrix(...)
G_csr = sot.emd(a, b, M_csr)

# POT-compatible log dict (cost, u, v, warning, result_code).
G, info = sot.emd(a, b, M, log=True)
```

The `u` and `v` returned in the log dict are the dual potentials with
POT's sign convention (`u[i] + v[j] ≤ M[i,j]` at the optimum). With
`center_dual=True` (default) `u` is shifted to zero mean, preserving
`u[i] + v[j]` on every edge.

## Why sparse?

A 10 000 × 10 000 problem with 10 candidate edges per row (k = 100 000):

| Solver path        | Memory        | Wall time |
|--------------------|---------------|-----------|
| Bonneel-dense      | ≈ 800 MB (cost matrix) | (does not run; OOM at this scale on small machines) |
| **Bonneel-sparse** | **≈ 6 MB**    | seconds   |

Most real OT problems (k-NN, transformer attention masks, point-cloud
matching) are intrinsically sparse. Materialising them as dense costs
matrices is wasteful and can be infeasible. This package gives you
Bonneel's tight constants without the O(n·m) memory penalty.

## Convergence and the `numItermax` knob

Bonneel's network simplex stops at `numItermax` pivots without raising.
If the iteration cap is hit before convergence the returned flow can
violate marginals by orders of magnitude more than machine epsilon. We
guard against this in two ways:

1. The default `numItermax` is **problem-size-aware**:
   `min(50M, max(100k, 100·(n + m + k)))`.
2. After every solve we re-check the marginals. If `max(|G.sum(1) - a|,
   |G.sum(0) - b|) > 1e-6`, we emit a `RuntimeWarning` and report
   `result_code = 0` with a diagnostic in `info["warning"]`. No
   exception is raised, matching POT's behavior.

You can pass `numItermax=…` to override.

## Build and install

```bash
pip install -e . --no-build-isolation
```

`pyproject.toml` sets `editable.rebuild = true`, so the pybind11
extension is rebuilt automatically the next time `sparse_ot` is imported
after a `src/cpp/` edit.

## Benchmarks

Two independent suites:

```bash
python benchmarks/bench_solvers.py --mid     # ~15 minutes
python benchmarks/bench_solvers.py --quick   # ~30 seconds (used by CI)
python benchmarks/bench_solvers.py           # full sweep, hours
```

- **Dense suite** — fully-random `n × n` cost matrix. Compares
  `bonneel_dense` against `pot_reference` (`ot.emd`).
- **Sparse suite** — feasible-by-construction kNN-grid (`benchmarks/problems.py`).
  Runs `bonneel_sparse` only; there is no honest dense representation of
  a kNN cost (absent edges must be +∞, which neither POT nor Bonneel's
  dense path can express).

Results are written to `benchmarks/results/{efficiency,accuracy}_{tag}.json`
with the structure:

```jsonc
{
  "dense":  { "<n>": { "bonneel_dense": {...}, "pot_reference": {...} } },
  "sparse": { "<n>": { "<k>": { "bonneel_sparse": {...} } } }
}
```

See "Benchmark results" below for the current numbers on the maintainer's
laptop.

## Benchmark results

Numbers below are from `python benchmarks/bench_solvers.py --mid` on an
Apple-Silicon laptop (Sonoma, 64 GB). Wall times are median of 1 run; the
sparse-suite peak memory is `tracemalloc` peak during `emd()`.

### Dense suite (fully random `n × n` cost)

| n     | `bonneel_dense` | `pot_reference` | dense / pot |
|------:|----------------:|----------------:|------------:|
|   200 |          0.003s |          0.002s |       0.66× |
|   500 |          0.019s |          0.015s |       0.80× |
|  1000 |          0.085s |          0.072s |       0.85× |
|  2000 |          0.406s |          0.346s |       0.85× |
|  4000 |          2.166s |          1.414s |       0.65× |

POT and `bonneel_dense` share the same C++ engine (POT vendors Bonneel's
network simplex), so the wall-time ratio reflects pure wrapping overhead;
POT's Cython wrapper is marginally tighter than our pybind11 wrapper.
Costs agree to machine precision for n ≤ 2000. At n = 4000, POT's cost
is 1.1 % higher than ours — POT's default `numItermax = 100 000`
truncates before convergence, while our problem-size-aware default
finishes the pivots.

### Sparse suite (knn-grid)

The same knn problem is run through both Bonneel paths so the
speed/memory tradeoff is directly comparable. `bonneel_dense` runs only
where `n ≤ MAX_DENSE_N`; above that the cost matrix doesn't fit and the
cell is sparse-only. For the dense path we densify with a finite
penalty (`max(M.data) · (n·m + 1)`) on absent edges — with the
problem-size-aware `numItermax` the optimal basis never lands on a
penalty edge.

| n      | k    | nnz        | sparse wall | sparse peak | dense wall | dense peak | winner       |
|-------:|-----:|-----------:|------------:|------------:|-----------:|-----------:|:-------------|
|    200 |    2 |        400 |      0.006s |    0.04 MB |     0.003s |    0.32 MB |  dense 2.4×  |
|    200 |   32 |      6 400 |      0.051s |    0.14 MB |     0.004s |    0.32 MB |  dense 14×   |
|    200 |  128 |     25 600 |      0.204s |    0.52 MB |     0.003s |    0.32 MB |  dense 71×   |
|  1 000 |    2 |      2 000 |      0.033s |    0.15 MB |     0.075s |    7.69 MB |  **sparse 2.3×** |
|  1 000 |    8 |      8 000 |      0.100s |    0.22 MB |     0.178s |    7.69 MB |  **sparse 1.8×** |
|  1 000 |   32 |     32 000 |      0.315s |    0.67 MB |     0.182s |    7.69 MB |  dense 1.7×  |
|  1 000 |  128 |    128 000 |      1.103s |    2.59 MB |     0.256s |    7.69 MB |  dense 4.3×  |
|  4 000 |    2 |      8 000 |      0.134s |    0.60 MB |     1.652s |  122.3  MB |  **sparse 12×**  |
|  4 000 |    8 |     32 000 |      0.577s |    0.87 MB |     8.918s |  122.3  MB |  **sparse 15×**  |
|  4 000 |   32 |    128 000 |      1.572s |    2.66 MB |     8.574s |  122.3  MB |  **sparse 5.5×** |
|  4 000 |  128 |    512 000 |      4.540s |   10.35 MB |    10.298s |  122.3  MB |  **sparse 2.3×** |
|  4 000 |  400 |  1 600 000 |     14.473s |   32.14 MB |     8.393s |  122.3  MB |  dense 1.7×  |
|  4 000 |  512 |  2 048 000 |     17.956s |   41.11 MB |     9.535s |  122.3  MB |  dense 1.9×  |
| 16 000 |    2 |     32 000 |      0.531s |    2.39 MB |       —    |      —     |  sparse only |
| 16 000 |   32 |    512 000 |     12.066s |   10.62 MB |       —    |      —     |  sparse only |
| 16 000 |  128 |  2 048 000 |     32.955s |   41.38 MB |       —    |      —     |  sparse only |
| 16 000 |  512 |  8 192 000 |     75.201s |  164.4  MB |       —    |      —     |  sparse only |
| 16 000 | 1600 | 25 600 000 |    269.488s |  513.1  MB |       —    |      —     |  sparse only |

Reading the table:

- At **n = 200** the dense path always wins because the n² cost matrix is
  tiny and Bonneel's flat-array constants dominate over the sparse
  digraph's per-arc indirection.
- At **n = 1 000** the crossover is around k ≈ n / 50: below that, sparse
  wins; above, dense wins.
- At **n = 4 000** sparse wins by 2–15× up to k ≈ n / 20. Above that
  density, dense again wins on time but its memory cost is fixed at
  122 MB regardless of k.
- At **n = 16 000** the dense path is out of reach (cost matrix ≈ 2 GB);
  only sparse runs.

### Accuracy

Marginals stay at machine precision (worst case `2.5 × 10⁻¹⁶`) across
every cell of both suites. On the knn problems, `bonneel_sparse` and
`bonneel_dense` agree on cost to ≈ machine precision in 22 of 24 cells.
Two exceptions:

| n     | k    | sparse cost | dense cost | relative diff |
|------:|-----:|------------:|-----------:|--------------:|
| 4 000 |  400 |  271.7297   | 272.2578   |    1.9 × 10⁻³ |
| 4 000 |  512 |  557.3260   | 557.3270   |    1.7 × 10⁻⁶ |

In both cases `bonneel_sparse` finds a strictly lower-cost plan. The
densified-with-penalty input introduces costs on the order of
`max(M) · n² ≈ 10¹²`, and floating-point reduced-cost computations on
that scale accumulate enough rounding noise to push the pivot rule off
the true optimum. The sparse path never sees those large numbers and is
the more accurate of the two when both can run.

## Memory cutoffs

`bench_solvers.py` skips cells beyond these defaults (16 GB target):

| Constant         | Default       | Effect                                     |
|------------------|---------------|--------------------------------------------|
| `MAX_DENSE_N`    | 8 192         | Dense suite skipped above this             |
| `MAX_SPARSE_NNZ` | 200 000 000   | Sparse cell skipped above this nnz         |

Raise the constants for larger hardware.

## Releasing

PyPI uploads are automated via GitHub Actions and PyPI's
[trusted-publishing OIDC](https://docs.pypi.org/trusted-publishers/). To
cut a release:

1. Bump `project.version` in `pyproject.toml`, commit, tag (`git tag vX.Y.Z`),
   push (`git push --tags`).
2. Create a GitHub Release pointing at the tag.

The `.github/workflows/publish.yml` workflow then builds wheels via
`cibuildwheel` for Linux (x86_64, arm64) and macOS (x86_64, arm64) across
Python 3.10–3.13, builds an sdist, and uploads everything to PyPI.

First-time setup (one-time, requires owner action on pypi.org):

- Add a trusted publisher for **sparse-ot** with owner = `JBobrutsky`,
  repository = `sparse-optimal-transport`, workflow = `publish.yml`,
  environment = `pypi`.
- For TestPyPI dry runs, register the same on test.pypi.org with
  environment = `testpypi`. Then trigger `Publish to PyPI` via the
  Actions UI (workflow_dispatch) with target = `testpypi`.

## License

MIT.
