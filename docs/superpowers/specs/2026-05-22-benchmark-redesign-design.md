# Benchmark Redesign & README Streamline — Design Spec

## Goal

Replace the two existing benchmark scripts and report generator with a single unified benchmark runner, unified JSON output schema, and a new report generator. Update the README to summarise results with compact tables and charts (Bonneel-style). Show that sparse-ot matches or beats POT (`ot.emd`) and OR-Tools (`min_cost_flow`) across all supported input types; validate correctness against POT up to machine precision; extrapolate competitor wall times at scales they cannot reach.

## Architecture

```
benchmarks/
  bench.py          ← unified orchestrator  (replaces bench_solvers.py + bench_refine.py)
  solvers.py        ← solver adapters with uniform interface
  report.py         ← chart + table generation  (replaces generate_report.py)
  problems.py       ← unchanged
  results/
    bench_quick.json
    bench_mid.json
    bench.json
  figures/
    dense_cold.png
    sparse_cold.png
    warm_speedup.png
    accuracy.png
```

`bench_solvers.py`, `bench_refine.py`, and `generate_report.py` are deleted.

---

## Benchmark Scenarios

Three scenarios; each produces rows in one flat `cells` list in the output JSON.

| Scenario | Problem | Solvers | Measured n | Extrapolated n |
|---|---|---|---|---|
| `dense_cold` | n×n random dense ndarray | sparse_ot, pot, ortools | 200–8 192 | beyond 8 192 |
| `sparse_cold` | kNN-grid CSR | sparse_ot (all n); pot+ortools (n ≤ 2 000) | 200–16 M (sparse_ot) | pot+ortools beyond 2 000 |
| `sparse_warm` | kNN-grid CSR, warm_ratio ∈ {0.25, 1.0} | sparse_ot only | 200–1 M | — |

Sweep sizes:

| Tag | Dense n | kNN n | kNN k | Runs |
|---|---|---|---|---|
| `--quick` | 200 | 200, 1 000 | 4, 32 | 1 |
| `--mid` | 200–4 000 | 200–16 000 | 2, 8, 32, 128, 512 | 1 |
| (full) | 200–8 192 | 200–16 M | 2, 8, 32, 128, 512, 2 048 | 5 |

---

## Solver Adapters (`solvers.py`)

Common return type:

```python
@dataclass
class SolveResult:
    cost: float
    wall_s: float
    peak_mb: float
    marginal_err_a: float   # max |G.sum(1) - a|
    marginal_err_b: float   # max |G.sum(0) - b|
```

### `sparse_ot`
Calls `emd(a, b, M, warm_start=warm, log=True)`. Accepts dense ndarray and CSR alike. `warm` is `(G, info)` from a prior solve; `None` for cold.

### `pot`
Calls `ot.emd(a, b, M)`. Dense ndarray only.
Skipped (returns `None`) when:
- `n > POT_MAX_N = 2 000`, or
- Input is CSR with `nnz > POT_MAX_NNZ = 100 000`.

### `ortools`
Calls `ortools.graph.python.min_cost_flow`. Requires integer costs and integer supplies.
Conversion: multiply costs by `SCALE = 1_000_000`, round to `int64`; multiply marginals by `SCALE` and round to integers summing to `SCALE`.
Accepts CSR (iterates nonzeros) or dense (iterates all entries).
Skipped when:
- `n > ORTOOLS_MAX_N = 2 000`, or
- `nnz > ORTOOLS_MAX_NNZ = 500 000`.

**Accuracy note:** integer scaling introduces rounding of order `1 / SCALE = 1e-6`. The correctness table reports this — OR-Tools agreement with sparse_ot is expected at `~1e-6`, not machine precision.

---

## Extrapolation

For each solver that has ≥ 3 measured data points, fit a power law in log-log space:

```
log(t) = log(a) + b·log(n) + c·log(k)     [sparse]
log(t) = log(a) + b·log(n)                 [dense, c omitted]
```

using `scipy.optimize.curve_fit`. Coefficients `(a, b, c)` and R² are stored in `fits` in the output JSON.

Extrapolated cells are flagged `"extrapolated": true` and rendered as **dashed lines** in charts.

The README shows extrapolated times with a `~` prefix (e.g. `~14 min`) and a footnote citing R². No extrapolation is shown if R² < 0.95 — those cells show `—`.

---

## Output JSON Schema

One flat file replaces all current split result files:

```jsonc
{
  "meta": {
    "host": "...", "timestamp": "...", "tag": "quick|mid|full",
    "cpu": "Apple M2", "python": "3.12"
  },
  "cells": [
    {
      "scenario": "sparse_cold",    // dense_cold | sparse_cold | sparse_warm
      "n": 1000,
      "k": 32,                      // null for dense scenarios
      "solver": "sparse_ot",        // sparse_ot | pot | ortools
      "warm_ratio": null,           // 0.25 | 1.0 for sparse_warm; null otherwise
      "wall_s": 0.315,
      "peak_mb": 0.67,
      "cost": 42.17,
      "marginal_err_a": 1.2e-15,
      "marginal_err_b": 9.8e-16,
      "extrapolated": false
    }
  ],
  "fits": {
    "pot_dense":      { "a": 1.2e-8, "b": 2.31, "r2": 0.999 },
    "ortools_dense":  { "a": 3.1e-8, "b": 2.28, "r2": 0.997 },
    "pot_sparse":     { "a": 4.5e-9, "b": 2.18, "c": 0.79, "r2": 0.995 },
    "ortools_sparse": { "a": 8.2e-9, "b": 2.22, "c": 0.81, "r2": 0.993 }
  }
}
```

`report.py` reads this single file to produce all charts and README table snippets.

---

## Report Generator (`report.py`)

Produces four figures and (optionally) prints Markdown table snippets:

| Figure | Type | Content |
|---|---|---|
| `dense_cold.png` | Line plot (log-log) | Wall time vs n: sparse_ot, pot, ortools; dashed = extrapolated |
| `sparse_cold.png` | Heatmap | Speedup ratio sparse_ot / pot at each (n, k); dashed contour at 1× |
| `warm_speedup.png` | Line plot (log-log) | Cold vs warm wall time vs n at warm_ratio=0.25 |
| `accuracy.png` | Scatter | Relative cost error vs POT per cell, grouped by scenario |

CLI:
```bash
python benchmarks/report.py --quick   # consume bench_quick.json
python benchmarks/report.py --mid
python benchmarks/report.py           # consume bench.json
python benchmarks/report.py --print-tables   # also print Markdown tables to stdout
```

---

## README Restructure

Target: ~350 lines. Benchmark section replaces current wall of text with four blocks.

### Block 1 — Dense cold

`![dense cold](benchmarks/results/figures/dense_cold.png)`

Compact table: n | sparse_ot | pot | ortools | vs-pot speedup. Extrapolated rows marked `~`.

### Block 2 — Sparse cold

`![sparse cold](benchmarks/results/figures/sparse_cold.png)`

Sub-table of extrapolated POT/OR-Tools times at large n. Footnote: power-law fit parameters and R².

### Block 3 — Warm-start speedup

`![warm speedup](benchmarks/results/figures/warm_speedup.png)`

Table: n | cold wall_s | warm wall_s | speedup. Caption: what warm_ratio=0.25 means.

### Block 4 — Correctness

Table: scenario | solver | max relative cost error | max marginal error. Caption explaining OR-Tools 1e-6 bound.

All other README sections (quickstart, feasibility, warm-start API, convergence, memory cutoffs, release instructions) are preserved and lightly edited for concision.

---

## Files Changed

| File | Action |
|---|---|
| `benchmarks/bench.py` | Create |
| `benchmarks/solvers.py` | Create |
| `benchmarks/report.py` | Create |
| `benchmarks/bench_solvers.py` | Delete |
| `benchmarks/bench_refine.py` | Delete |
| `benchmarks/generate_report.py` | Delete |
| `benchmarks/problems.py` | No change |
| `benchmarks/results/bench_quick.json` | Create (new schema) |
| `README.md` | Modify benchmark section |

---

## Testing

- `python benchmarks/bench.py --quick` completes without error and writes `bench_quick.json` with the correct schema.
- `python benchmarks/report.py --quick` produces all four PNGs without error.
- All existing `pytest tests/` pass (benchmark changes do not touch `src/`).
- Correctness: for all measured sparse_ot vs pot cells, `|cost_ours - cost_pot| / cost_pot < 1e-10`.
