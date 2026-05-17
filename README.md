# sparse-ot

Drop-in replacement for [POT](https://github.com/PythonOT/POT)'s `emd` / `emd2`,
optimized for sparse cost matrices. Routes between three solvers based on
problem size and sparsity:

| Solver       | Best for                                       | Source             |
|--------------|------------------------------------------------|--------------------|
| Bonneel      | Dense / near-dense cost matrices               | Vendored C++ (`src/cpp/bonneel`) |
| LEMON        | Sparse, moderate scale (float64-patched)       | Vendored C++ (`src/cpp/lemon`) + custom patch |
| OR-Tools     | Very large sparse problems                     | Optional `ortools` Python dep |

> **Editable installs auto-rebuild on import.** `pyproject.toml` sets
> `editable.rebuild = true` so the pybind11 extensions are recompiled
> automatically the next time `sparse_ot` is imported after a `src/cpp/`
> edit. The persistent build directory is `build/{wheel_tag}/`. To force
> a clean rebuild manually, run
> `uv pip install --no-build-isolation -e . --force-reinstall --no-deps`.

## Quickstart

```python
import sparse_ot as sot
G = sot.emd(a, b, M)        # numpy dense in → numpy dense out
G = sot.emd(a, b, M_csr)    # scipy CSR in → scipy CSR out
cost = sot.emd2(a, b, M)
```

`solver=` overrides routing: `'bonneel'`, `'lemon'`, or `'ortools'`.
`cost_sparsity_threshold` drops edges with `|M[i,j]| <= threshold` from dense
input. `ortools_cost_scale` (default `1e6`) controls int64 cost precision for
the OR-Tools backend.

## Routing

`routing.select_solver(n, m, nnz, solver=None)` picks the solver from
`benchmarks/results/routing_thresholds.json`:

- `k = nnz / n > bonneel_lemon` → Bonneel
- otherwise `n > lemon_ortools` → OR-Tools
- otherwise → LEMON

The thresholds are derived empirically from the benchmark suite. To regenerate
for your hardware:

```bash
python benchmarks/bench_solvers.py        # full sweep — hours
python benchmarks/generate_report.py      # writes routing_thresholds.json + figures
```

For development, the `--quick` flag runs a small sweep in seconds:

```bash
python benchmarks/bench_solvers.py --quick
python benchmarks/generate_report.py --quick    # renders figures only; does not touch routing_thresholds.json
```

## Memory cutoffs

`bench_solvers.py` skips cells beyond these defaults (16 GB target):

| Constant         | Default       | Effect                              |
|------------------|---------------|-------------------------------------|
| `MAX_DENSE_N`    | 8 192         | Bonneel and POT reference skipped above this |
| `MAX_SPARSE_NNZ` | 200 000 000   | LEMON skipped above this            |
| `MAX_ORTOOLS_NNZ`| 500 000 000   | OR-Tools skipped above this         |

The reference voxel-grid case (n = 16.7M, nnz ≈ 536M) exceeds `MAX_ORTOOLS_NNZ`
on a 16GB machine. Raise the constants in `benchmarks/bench_solvers.py` for
larger hardware.

## Install

```bash
pip install sparse-ot                 # numpy + scipy
pip install sparse-ot[ortools]        # adds OR-Tools solver
pip install sparse-ot[torch]          # adds torch sparse interop
```

## License

MIT.
