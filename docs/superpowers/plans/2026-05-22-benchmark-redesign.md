# Benchmark Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two legacy benchmark scripts and a report generator with a unified runner (`bench.py`), solver-adapter module (`solvers.py`), and report generator (`report.py`) that proves sparse-ot is competitive with POT and OR-Tools across all supported input types, with machine-precision correctness and power-law-extrapolated competitor wall times at scales they cannot reach.

**Architecture:** `solvers.py` owns a `SolveResult` dataclass and three adapter functions (`solve_sparse_ot`, `solve_pot`, `solve_ortools`). `bench.py` orchestrates three scenarios (`dense_cold`, `sparse_cold`, `sparse_warm`), writes a flat `cells` list plus power-law `fits` dict to `benchmarks/results/bench_{tag}.json`. `report.py` reads that single file and produces four PNG figures. `README.md` benchmark section is rewritten to embed the PNGs with compact tables.

**Tech Stack:** Python 3.10+, numpy, scipy (`curve_fit`), matplotlib, POT (`ot.emd`), OR-Tools (`ortools.graph.python.min_cost_flow`), `sparse_ot.emd`, `benchmarks/problems.py`.

**Spec:** `docs/superpowers/specs/2026-05-22-benchmark-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `benchmarks/solvers.py` | Create | SolveResult + 3 adapter functions |
| `benchmarks/bench.py` | Create | Sweep orchestrator, JSON output |
| `benchmarks/report.py` | Create | 4 PNG figures |
| `benchmarks/bench_solvers.py` | Delete | (replaced by bench.py) |
| `benchmarks/bench_refine.py` | Delete | (replaced by bench.py) |
| `benchmarks/generate_report.py` | Delete | (replaced by report.py) |
| `benchmarks/problems.py` | No change | — |
| `tests/test_solvers.py` | Create | Unit tests for solver adapters |
| `tests/test_benchmarks.py` | Modify | Update smoke test for new JSON schema |
| `pyproject.toml` | Modify | Add `ortools` to `[dev]` extras |

---

## Task 1: `solvers.py` — SolveResult dataclass and three adapter functions

**Files:**
- Create: `benchmarks/solvers.py`
- Create: `tests/test_solvers.py`
- Modify: `pyproject.toml` (add `ortools` to dev extras)

### Background

The spec defines a uniform return type:

```
SolveResult(cost, wall_s, peak_mb, marginal_err_a, marginal_err_b)
```

Skip thresholds (adapters return `None` without running):
- `pot`: skip when `n > POT_MAX_N = 2_000` or CSR `nnz > POT_MAX_NNZ = 100_000`
- `ortools`: skip when `n > ORTOOLS_MAX_N = 2_000` or `nnz > ORTOOLS_MAX_NNZ = 500_000`

OR-Tools requires integer costs: multiply by `SCALE = 1_000_000`, round.  
Accuracy from OR-Tools is ~1e-6 (not machine precision).

- [ ] **Step 1: Write the tests (they will fail — `solvers.py` does not exist yet)**

Create `tests/test_solvers.py`:

```python
"""Unit tests for benchmarks/solvers.py adapter functions."""
import numpy as np
import pytest
import scipy.sparse


def _two_by_two():
    """2×2 OT problem with known optimal cost = 1.0."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    M = np.array([[1.0, 2.0], [3.0, 1.0]])
    return a, b, M


def test_solve_result_fields():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    r = solve_sparse_ot(a, b, M)
    assert hasattr(r, "cost")
    assert hasattr(r, "wall_s")
    assert hasattr(r, "peak_mb")
    assert hasattr(r, "marginal_err_a")
    assert hasattr(r, "marginal_err_b")


def test_sparse_ot_dense_correct():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    r = solve_sparse_ot(a, b, M)
    assert abs(r.cost - 1.0) < 1e-9
    assert r.marginal_err_a < 1e-10
    assert r.marginal_err_b < 1e-10
    assert r.wall_s > 0
    assert r.peak_mb >= 0


def test_sparse_ot_csr_correct():
    from benchmarks.solvers import solve_sparse_ot
    a, b, M = _two_by_two()
    M_csr = scipy.sparse.csr_matrix(M)
    r = solve_sparse_ot(a, b, M_csr)
    assert abs(r.cost - 1.0) < 1e-9


def test_pot_agrees_with_sparse_ot():
    from benchmarks.solvers import solve_sparse_ot, solve_pot
    a, b, M = _two_by_two()
    r_ot = solve_sparse_ot(a, b, M)
    r_pot = solve_pot(a, b, M)
    assert r_pot is not None
    assert abs(r_pot.cost - r_ot.cost) < 1e-10
    assert r_pot.marginal_err_a < 1e-10
    assert r_pot.marginal_err_b < 1e-10


def test_pot_skips_large_n():
    from benchmarks.solvers import solve_pot, POT_MAX_N
    n = POT_MAX_N + 1
    # Diagonal CSR: n edges but n > POT_MAX_N triggers skip.
    M = scipy.sparse.eye(n, format="csr")
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_pot(a, b, M) is None


def test_pot_skips_large_nnz():
    from benchmarks.solvers import solve_pot, POT_MAX_NNZ
    # 317×317 = 100489 > 100000.
    n = 317
    M = scipy.sparse.csr_matrix(np.ones((n, n)))
    assert M.nnz > POT_MAX_NNZ
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_pot(a, b, M) is None


def test_ortools_agrees_with_sparse_ot():
    from benchmarks.solvers import solve_sparse_ot, solve_ortools
    a, b, M = _two_by_two()
    r_ot = solve_sparse_ot(a, b, M)
    r_or = solve_ortools(a, b, M)
    assert r_or is not None
    # OR-Tools rounds costs to 1/SCALE; expect ~1e-6 relative accuracy.
    assert abs(r_or.cost - r_ot.cost) < 1e-5
    assert r_or.marginal_err_a < 2e-6
    assert r_or.marginal_err_b < 2e-6


def test_ortools_skips_large_nnz():
    from benchmarks.solvers import solve_ortools, ORTOOLS_MAX_NNZ
    # 800×800 = 640000 > 500000.
    n = 800
    M = scipy.sparse.csr_matrix(np.ones((n, n)))
    assert M.nnz > ORTOOLS_MAX_NNZ
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_ortools(a, b, M) is None


def test_ortools_skips_large_n():
    from benchmarks.solvers import solve_ortools, ORTOOLS_MAX_N
    n = ORTOOLS_MAX_N + 1
    M = scipy.sparse.eye(n, format="csr")
    a = np.ones(n) / n
    b = np.ones(n) / n
    assert solve_ortools(a, b, M) is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_solvers.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'benchmarks.solvers'`

- [ ] **Step 3: Add `ortools` to dev deps**

In `pyproject.toml`, change:

```toml
dev = ["pytest>=7.0", "pytest-timeout>=2.0", "pot>=0.9", "pybind11>=2.12", "matplotlib>=3.7"]
```

to:

```toml
dev = ["pytest>=7.0", "pytest-timeout>=2.0", "pot>=0.9", "pybind11>=2.12", "matplotlib>=3.7", "ortools>=9.0", "scipy>=1.10"]
```

- [ ] **Step 4: Create `benchmarks/solvers.py`**

```python
"""Solver adapters with a uniform SolveResult interface.

Each adapter returns SolveResult or None (when the problem exceeds a
size cutoff). None means "skip" — the caller records no cell.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import scipy.sparse

# ── skip thresholds ──────────────────────────────────────────────────────────
POT_MAX_N        = 2_000
POT_MAX_NNZ      = 100_000
ORTOOLS_MAX_N    = 2_000
ORTOOLS_MAX_NNZ  = 500_000

SCALE = 1_000_000   # integer factor for OR-Tools cost/supply conversion


@dataclass
class SolveResult:
    cost: float
    wall_s: float
    peak_mb: float
    marginal_err_a: float   # max |G.sum(1) - a|
    marginal_err_b: float   # max |G.sum(0) - b|


def _measure(fn):
    """Call fn(), return (return_value, wall_s_float, peak_mb_float)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, wall_s, peak_bytes / 1024 ** 2


def _marginal_errors(G, a, b):
    if scipy.sparse.issparse(G):
        row_sum = np.asarray(G.sum(axis=1)).ravel()
        col_sum = np.asarray(G.sum(axis=0)).ravel()
    else:
        row_sum = np.asarray(G).sum(axis=1)
        col_sum = np.asarray(G).sum(axis=0)
    return float(np.abs(row_sum - a).max()), float(np.abs(col_sum - b).max())


def solve_sparse_ot(a, b, M, warm=None) -> SolveResult:
    """Run sparse_ot.emd. Accepts dense ndarray or CSR. warm=(G, info) or None."""
    from sparse_ot import emd

    (G, info), wall_s, peak_mb = _measure(lambda: emd(a, b, M, warm_start=warm, log=True))
    err_a, err_b = _marginal_errors(G, a, b)
    return SolveResult(
        cost=info["cost"],
        wall_s=wall_s,
        peak_mb=peak_mb,
        marginal_err_a=err_a,
        marginal_err_b=err_b,
    )


def solve_pot(a, b, M) -> SolveResult | None:
    """Run ot.emd on a dense ndarray. Returns None when n or nnz exceeds cutoff."""
    import ot

    if scipy.sparse.issparse(M):
        if M.nnz > POT_MAX_NNZ:
            return None
        n = M.shape[0]
        M_dense = M.toarray().astype(np.float64)
    else:
        M_dense = np.asarray(M, dtype=np.float64)
        n = M_dense.shape[0]

    if n > POT_MAX_N:
        return None

    G, wall_s, peak_mb = _measure(lambda: ot.emd(a, b, M_dense))
    cost = float(np.sum(G * M_dense))
    err_a, err_b = _marginal_errors(G, a, b)
    return SolveResult(cost=cost, wall_s=wall_s, peak_mb=peak_mb,
                       marginal_err_a=err_a, marginal_err_b=err_b)


def solve_ortools(a, b, M) -> SolveResult | None:
    """Run OR-Tools min_cost_flow. Costs rounded to integers (accuracy ~1e-6).

    Returns None when n or nnz exceeds cutoff.
    Accepts dense ndarray or CSR.
    """
    from ortools.graph.python import min_cost_flow as mcf_mod

    if scipy.sparse.issparse(M):
        M_csr = M.tocsr()
        n = M_csr.shape[0]
        nnz = M_csr.nnz
    else:
        M_arr = np.asarray(M, dtype=np.float64)
        n = M_arr.shape[0]
        nnz = M_arr.size

    if n > ORTOOLS_MAX_N or nnz > ORTOOLS_MAX_NNZ:
        return None

    # Integer supplies summing to SCALE; fix rounding imbalance on supply_b[0].
    supply_a = np.round(a * SCALE).astype(np.int64)
    supply_b = np.round(b * SCALE).astype(np.int64)
    supply_b[0] += supply_a.sum() - supply_b.sum()

    smcf = mcf_mod.SimpleMinCostFlow()
    # Nodes: sources 0..n-1, sinks n..2n-1.
    if scipy.sparse.issparse(M):
        coo = M_csr.tocoo()
        for i, j, c in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
            smcf.add_arc_with_capacity_and_unit_cost(
                int(i), n + int(j), SCALE, int(round(c * SCALE))
            )
    else:
        for i in range(n):
            for j in range(n):
                smcf.add_arc_with_capacity_and_unit_cost(
                    i, n + j, SCALE, int(round(M_arr[i, j] * SCALE))
                )

    for i in range(n):
        smcf.set_node_supply(i, int(supply_a[i]))
    for j in range(n):
        smcf.set_node_supply(n + j, -int(supply_b[j]))

    status, wall_s, peak_mb = _measure(smcf.solve)

    if status != smcf.OPTIMAL:
        return None

    cost = smcf.optimal_cost() / (SCALE * SCALE)

    # Reconstruct row/col sums from arc flows to compute marginal errors.
    row_flow = np.zeros(n, dtype=np.float64)
    col_flow = np.zeros(n, dtype=np.float64)
    for arc in range(smcf.num_arcs()):
        flow = smcf.flow(arc) / SCALE
        row_flow[smcf.tail(arc)] += flow
        col_flow[smcf.head(arc) - n] += flow

    return SolveResult(
        cost=cost,
        wall_s=wall_s,
        peak_mb=peak_mb,
        marginal_err_a=float(np.abs(row_flow - a).max()),
        marginal_err_b=float(np.abs(col_flow - b).max()),
    )
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_solvers.py -v
```

Expected output: 9 tests pass (PASSED for all).

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all existing tests pass plus the 9 new ones.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/solvers.py tests/test_solvers.py pyproject.toml
git commit -m "feat(bench): solvers.py — SolveResult + sparse_ot/pot/ortools adapters"
```

---

## Task 2: `bench.py` — orchestrator skeleton + dense_cold + sparse_cold scenarios

**Files:**
- Create: `benchmarks/bench.py`

### Background

`bench.py` produces a single JSON file with this top-level structure:

```jsonc
{
  "meta": {"host": "...", "timestamp": "...", "tag": "quick|mid|full", "cpu": "...", "python": "3.12"},
  "cells": [ /* flat list, one dict per measured or extrapolated cell */ ],
  "fits":  { /* power-law fit params — added in Task 4 */ }
}
```

Each measured cell:

```jsonc
{
  "scenario": "dense_cold",    // dense_cold | sparse_cold | sparse_warm
  "n": 1000,
  "k": null,                   // null for dense_cold
  "solver": "sparse_ot",       // sparse_ot | pot | ortools
  "warm_ratio": null,          // float for sparse_warm, null otherwise
  "wall_s": 0.085,
  "peak_mb": 7.69,
  "cost": 42.17,
  "marginal_err_a": 1.2e-15,
  "marginal_err_b": 9.8e-16,
  "extrapolated": false
}
```

Sweep definitions (this task implements `--quick` and `--mid`; `--quick` is the one CI checks):

```
--quick : DENSE_NS=[200], KNN_NS=[200, 1_000], KNN_KS=[4, 32], RUNS=1
--mid   : DENSE_NS=[200,500,1000,2000,4000], KNN_NS=[200,1000,4000,16000], KNN_KS=[2,8,32,128,512], RUNS=1
full    : DENSE_NS=[200,500,1000,2000,4000,8192], KNN_NS=[200,1000,4000,16000,64000,256000,1000000,4000000,16000000], KNN_KS=[2,8,32,128,512,2048], RUNS=5
```

**In this task** implement:
1. `dense_cold` scenario: `generate_dense_random_problem` × [sparse_ot, pot, ortools]
2. `sparse_cold` scenario: `generate_knn_grid_problem` × [sparse_ot, pot (n≤2000), ortools (n≤2000)]
3. `fits` is written as `{}` (Task 4 adds real fits)
4. `--quick` smoke test passes

- [ ] **Step 1: Add a smoke test for bench.py --quick to `tests/test_benchmarks.py`**

Replace the existing `test_bench_solvers_quick_smoke` function with the one below. The old function tests the *deleted* `bench_solvers.py` output schema; the new one tests `bench.py`.

Find `def test_bench_solvers_quick_smoke` in `tests/test_benchmarks.py` and replace the entire function body:

```python
@pytest.mark.timeout(600)
def test_bench_quick_smoke(tmp_path):
    """`python benchmarks/bench.py --quick` exits 0 and writes bench_quick.json."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "benchmarks/bench.py", "--quick"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"bench.py --quick failed:\n{result.stderr}"

    out_path = repo_root / "benchmarks/results/bench_quick.json"
    assert out_path.exists(), "bench_quick.json not written"

    data = json.loads(out_path.read_text())
    assert "meta" in data
    assert "cells" in data
    assert "fits" in data
    assert isinstance(data["cells"], list)
    assert len(data["cells"]) > 0

    # Every cell must have the required schema keys.
    required = {"scenario", "n", "k", "solver", "warm_ratio",
                "wall_s", "peak_mb", "cost", "marginal_err_a", "marginal_err_b",
                "extrapolated"}
    for cell in data["cells"]:
        missing = required - set(cell.keys())
        assert not missing, f"Cell missing keys {missing}: {cell}"

    # Must contain dense_cold and sparse_cold cells with sparse_ot solver.
    scenarios = {c["scenario"] for c in data["cells"]}
    assert "dense_cold" in scenarios
    assert "sparse_cold" in scenarios

    # sparse_ot runs for all cells.
    sparse_ot_cells = [c for c in data["cells"] if c["solver"] == "sparse_ot"]
    assert len(sparse_ot_cells) > 0

    # dense_cold cells must have k=null.
    for c in data["cells"]:
        if c["scenario"] == "dense_cold":
            assert c["k"] is None

    # sparse_cold cells must have k set.
    for c in data["cells"]:
        if c["scenario"] == "sparse_cold":
            assert c["k"] is not None
```

- [ ] **Step 2: Run the updated test to confirm it fails (bench.py doesn't exist yet)**

```bash
pytest tests/test_benchmarks.py::test_bench_quick_smoke -v
```

Expected: FAILED (FileNotFoundError or returncode != 0)

- [ ] **Step 3: Create `benchmarks/bench.py`**

```python
"""Unified benchmark runner — replaces bench_solvers.py + bench_refine.py.

Usage:
    python benchmarks/bench.py --quick    # ~30 s; writes bench_quick.json
    python benchmarks/bench.py --mid      # ~15 min; writes bench_mid.json
    python benchmarks/bench.py            # full sweep; writes bench.json

The output JSON has this top-level shape:
    {"meta": {...}, "cells": [...], "fits": {...}}

Each cell is one solver × one problem instance. See spec for full schema.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── repo-root path fix ──────────────────────────────────────────────────────
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.problems import generate_dense_random_problem, generate_knn_grid_problem
from benchmarks.solvers import solve_sparse_ot, solve_pot, solve_ortools

RESULTS_DIR = Path(__file__).parent / "results"

# ── sweep parameters ─────────────────────────────────────────────────────────
DENSE_NS_QUICK = [200]
DENSE_NS_MID   = [200, 500, 1_000, 2_000, 4_000]
DENSE_NS_FULL  = [200, 500, 1_000, 2_000, 4_000, 8_192]

KNN_NS_QUICK   = [200, 1_000]
KNN_KS_QUICK   = [4, 32]

KNN_NS_MID     = [200, 1_000, 4_000, 16_000]
KNN_KS_MID     = [2, 8, 32, 128, 512]

KNN_NS_FULL    = [200, 1_000, 4_000, 16_000, 64_000, 256_000,
                  1_000_000, 4_000_000, 16_000_000]
KNN_KS_FULL    = [2, 8, 32, 128, 512, 2_048]

KNN_NS_WARM_MAX = 1_000_000   # warm scenario capped at 1 M
WARM_RATIOS     = [0.25, 1.0]


# ── helpers ──────────────────────────────────────────────────────────────────

def _cell(scenario, n, k, solver, warm_ratio, result, extrapolated=False):
    return {
        "scenario":      scenario,
        "n":             n,
        "k":             k,
        "solver":        solver,
        "warm_ratio":    warm_ratio,
        "wall_s":        result.wall_s,
        "peak_mb":       result.peak_mb,
        "cost":          result.cost,
        "marginal_err_a": result.marginal_err_a,
        "marginal_err_b": result.marginal_err_b,
        "extrapolated":  extrapolated,
    }


def _meta(tag):
    cpu = platform.processor() or platform.machine()
    return {
        "host":      socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag":       tag,
        "cpu":       cpu,
        "python":    f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _restrict_to_k(M_full_csr, k_warm):
    """Sub-select the k_warm cheapest edges per row from M_full's support."""
    import scipy.sparse
    n = M_full_csr.shape[0]
    indptr  = M_full_csr.indptr
    indices = M_full_csr.indices
    data    = M_full_csr.data

    keep_mask = np.zeros(len(data), dtype=bool)
    for i in range(n):
        s, e = int(indptr[i]), int(indptr[i + 1])
        row_len = e - s
        if row_len <= k_warm:
            keep_mask[s:e] = True
        else:
            local_keep = np.argpartition(data[s:e], k_warm)[:k_warm]
            keep_mask[s + local_keep] = True

    row_idx = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    return scipy.sparse.csr_matrix(
        (data[keep_mask], (row_idx[keep_mask], indices[keep_mask])),
        shape=M_full_csr.shape,
    )


# ── scenario runners ─────────────────────────────────────────────────────────

def run_dense_cold(dense_ns, runs):
    """dense_cold: fully-random n×n dense matrix, 3 solvers."""
    cells = []
    for n in dense_ns:
        a, b, M = generate_dense_random_problem(n, seed=0)
        print(f"  dense_cold n={n}", flush=True)

        for _ in range(runs):
            gc.collect()
            r = solve_sparse_ot(a, b, M)
            cells.append(_cell("dense_cold", n, None, "sparse_ot", None, r))

            r = solve_pot(a, b, M)
            if r is not None:
                cells.append(_cell("dense_cold", n, None, "pot", None, r))

            r = solve_ortools(a, b, M)
            if r is not None:
                cells.append(_cell("dense_cold", n, None, "ortools", None, r))

    return cells


def run_sparse_cold(knn_ns, knn_ks, runs):
    """sparse_cold: kNN-grid CSR, sparse_ot always; pot+ortools for n≤threshold."""
    cells = []
    for n in knn_ns:
        for k in knn_ks:
            if k > n:
                continue
            a, b, M, _ = generate_knn_grid_problem(n, k, seed=0)
            M = M.tocsr()
            print(f"  sparse_cold n={n} k={k} nnz={M.nnz}", flush=True)

            for _ in range(runs):
                gc.collect()
                r = solve_sparse_ot(a, b, M)
                cells.append(_cell("sparse_cold", n, k, "sparse_ot", None, r))

                r = solve_pot(a, b, M)
                if r is not None:
                    cells.append(_cell("sparse_cold", n, k, "pot", None, r))

                r = solve_ortools(a, b, M)
                if r is not None:
                    cells.append(_cell("sparse_cold", n, k, "ortools", None, r))

    return cells


def run_sparse_warm(knn_ns, knn_ks, runs):
    """sparse_warm: warm-start refinement, sparse_ot only, warm_ratio ∈ {0.25, 1.0}.

    wall_s records the refinement step only (phase-2), not the phase-1 cold
    solve on M_warm. The cold baseline comes from sparse_cold cells.
    """
    from sparse_ot import emd

    cells = []
    for n in knn_ns:
        if n > KNN_NS_WARM_MAX:
            continue
        for k in knn_ks:
            if k > n:
                continue
            a, b, M_full, _ = generate_knn_grid_problem(n, k, seed=0)
            M_full = M_full.tocsr()

            for warm_ratio in WARM_RATIOS:
                k_warm = max(2, int(round(k * warm_ratio)))
                M_warm = _restrict_to_k(M_full, k_warm)
                print(f"  sparse_warm n={n} k={k} warm_ratio={warm_ratio}", flush=True)

                for _ in range(runs):
                    gc.collect()
                    # Phase 1: cold solve on M_warm — not timed for the cell.
                    G_warm, info_warm = emd(a, b, M_warm, log=True)
                    # Phase 2: timed refinement on M_full.
                    r = solve_sparse_ot(a, b, M_full, warm=(G_warm, info_warm))
                    cells.append(_cell("sparse_warm", n, k, "sparse_ot", warm_ratio, r))

    return cells


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mid",   action="store_true")
    args = ap.parse_args()

    if args.quick:
        dense_ns, knn_ns, knn_ks, runs, tag = (
            DENSE_NS_QUICK, KNN_NS_QUICK, KNN_KS_QUICK, 1, "quick"
        )
    elif args.mid:
        dense_ns, knn_ns, knn_ks, runs, tag = (
            DENSE_NS_MID, KNN_NS_MID, KNN_KS_MID, 1, "mid"
        )
    else:
        dense_ns, knn_ns, knn_ks, runs, tag = (
            DENSE_NS_FULL, KNN_NS_FULL, KNN_KS_FULL, 5, "full"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix  = f"_{tag}" if tag != "full" else ""
    out_path = RESULTS_DIR / f"bench{suffix}.json"

    print(f"[bench] tag={tag} dense_ns={dense_ns}", flush=True)
    print(f"[bench] knn_ns={knn_ns} knn_ks={knn_ks} runs={runs}", flush=True)

    cells = []
    cells += run_dense_cold(dense_ns, runs)
    cells += run_sparse_cold(knn_ns, knn_ks, runs)
    cells += run_sparse_warm(knn_ns, knn_ks, runs)

    out = {"meta": _meta(tag), "cells": cells, "fits": {}}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path} ({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test**

```bash
pytest tests/test_benchmarks.py::test_bench_quick_smoke -v
```

Expected: PASSED (may take up to 2 minutes).

- [ ] **Step 5: Spot-check the output schema**

```bash
python -c "
import json
d = json.loads(open('benchmarks/results/bench_quick.json').read())
print('cells:', len(d['cells']))
print('scenarios:', {c['scenario'] for c in d['cells']})
print('solvers:', {c['solver'] for c in d['cells']})
print('sample cell:', d['cells'][0])
"
```

Expected: 
- `cells: 12` or more (dense_cold n=200 × 3 solvers + sparse_cold n=[200,1000] × k=[4,32] × solvers + warm)
- `scenarios: {'dense_cold', 'sparse_cold', 'sparse_warm'}`

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/bench.py tests/test_benchmarks.py
git commit -m "feat(bench): bench.py orchestrator — dense_cold + sparse_cold + sparse_warm"
```

---

## Task 3: `bench.py` — power-law extrapolation (fills `fits` key)

**Files:**
- Modify: `benchmarks/bench.py` (add `_compute_fits` and `_add_extrapolated_cells`)

### Background

For each solver that was skipped at some n (pot/ortools beyond their cutoffs), we fit a power law in log-log space:

- Dense: `log(t) = log(a) + b·log(n)` (2 params)
- Sparse: `log(t) = log(a) + b·log(n) + c·log(k)` (3 params)

Using `scipy.optimize.curve_fit` on the measured cells (≥ 3 data points required).  
Extrapolated cells are added for n/k values in the sweep that the solver didn't measure, flagged `"extrapolated": true`. If R² < 0.95 or fewer than 3 data points exist, no fit is stored and no extrapolated cells are added.

Fit keys: `"pot_dense"`, `"ortools_dense"`, `"pot_sparse"`, `"ortools_sparse"`.

- [ ] **Step 1: Write a test for the fits output**

Add to `tests/test_benchmarks.py`:

```python
@pytest.mark.timeout(600)
def test_bench_quick_fits_structure(tmp_path):
    """After bench.py --quick, the fits key exists (may be empty for quick sweep)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "benchmarks/results/bench_quick.json"
    # Re-use existing output if fresh enough; otherwise re-run.
    if not out_path.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, "benchmarks/bench.py", "--quick"],
            cwd=str(repo_root), env=env, check=True, timeout=600,
        )
    data = json.loads(out_path.read_text())
    assert isinstance(data["fits"], dict)
    # If fits are present, each entry must have at least a, b, r2.
    for key, val in data["fits"].items():
        assert "a" in val and "b" in val and "r2" in val, f"bad fit for {key}: {val}"
        assert 0.0 <= val["r2"] <= 1.0
    # Extrapolated cells must be flagged.
    for c in data["cells"]:
        assert "extrapolated" in c
        assert isinstance(c["extrapolated"], bool)
```

- [ ] **Step 2: Run the new test (it should pass trivially since fits={} is already valid)**

```bash
pytest tests/test_benchmarks.py::test_bench_quick_fits_structure -v
```

Expected: PASSED (fits is `{}` now, loop body never runs).

- [ ] **Step 3: Add `_compute_fits` and `_add_extrapolated_cells` to `bench.py`**

Add these two functions **above** `main()` in `benchmarks/bench.py`:

```python
def _compute_fits(cells):
    """Fit power-law models for pot and ortools from measured cells.

    Returns a dict with keys like "pot_dense", "ortools_sparse", etc.
    Only stored when R² >= 0.95 and ≥ 3 measured data points exist.
    """
    from scipy.optimize import curve_fit

    fits = {}

    def _fit_dense(solver):
        pts = [
            (c["n"], c["wall_s"])
            for c in cells
            if c["scenario"] == "dense_cold"
            and c["solver"] == solver
            and not c["extrapolated"]
        ]
        if len(pts) < 3:
            return None
        ns = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts], dtype=float)
        try:
            def model(log_n, log_a, b):
                return log_a + b * log_n
            popt, _ = curve_fit(model, np.log(ns), np.log(ts))
            log_a, b = popt
            y_pred = model(np.log(ns), *popt)
            ss_res = np.sum((np.log(ts) - y_pred) ** 2)
            ss_tot = np.sum((np.log(ts) - np.log(ts).mean()) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            if r2 < 0.95:
                return None
            return {"a": float(np.exp(log_a)), "b": float(b), "r2": r2}
        except Exception:
            return None

    def _fit_sparse(solver):
        pts = [
            (c["n"], c["k"], c["wall_s"])
            for c in cells
            if c["scenario"] == "sparse_cold"
            and c["solver"] == solver
            and not c["extrapolated"]
            and c["k"] is not None
        ]
        if len(pts) < 3:
            return None
        ns = np.array([p[0] for p in pts], dtype=float)
        ks = np.array([p[1] for p in pts], dtype=float)
        ts = np.array([p[2] for p in pts], dtype=float)
        try:
            def model(X, log_a, b, c_):
                return log_a + b * X[0] + c_ * X[1]
            popt, _ = curve_fit(model, (np.log(ns), np.log(ks)), np.log(ts))
            log_a, b, c_ = popt
            y_pred = model((np.log(ns), np.log(ks)), *popt)
            ss_res = np.sum((np.log(ts) - y_pred) ** 2)
            ss_tot = np.sum((np.log(ts) - np.log(ts).mean()) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            if r2 < 0.95:
                return None
            return {"a": float(np.exp(log_a)), "b": float(b), "c": float(c_), "r2": r2}
        except Exception:
            return None

    for solver in ("pot", "ortools"):
        f = _fit_dense(solver)
        if f:
            fits[f"{solver}_dense"] = f
        f = _fit_sparse(solver)
        if f:
            fits[f"{solver}_sparse"] = f

    return fits


def _add_extrapolated_cells(cells, fits, dense_ns, knn_ns, knn_ks):
    """Append extrapolated cells for n/k combinations the solver skipped."""
    from benchmarks.solvers import POT_MAX_N, ORTOOLS_MAX_N

    extra = []

    # Dense: add extrapolated pot/ortools cells at n > their cutoff.
    measured_dense = {
        (c["solver"], c["n"])
        for c in cells
        if c["scenario"] == "dense_cold" and not c["extrapolated"]
    }
    for n in dense_ns:
        for solver, cutoff, fit_key in [
            ("pot",     POT_MAX_N,     "pot_dense"),
            ("ortools", ORTOOLS_MAX_N, "ortools_dense"),
        ]:
            if (solver, n) in measured_dense:
                continue
            if fit_key not in fits:
                continue
            f = fits[fit_key]
            t_hat = f["a"] * n ** f["b"]
            extra.append({
                "scenario": "dense_cold", "n": n, "k": None,
                "solver": solver, "warm_ratio": None,
                "wall_s": t_hat, "peak_mb": None,
                "cost": None, "marginal_err_a": None, "marginal_err_b": None,
                "extrapolated": True,
            })

    # Sparse: add extrapolated pot/ortools cells at (n, k) they skipped.
    measured_sparse = {
        (c["solver"], c["n"], c["k"])
        for c in cells
        if c["scenario"] == "sparse_cold" and not c["extrapolated"]
    }
    for n in knn_ns:
        for k in knn_ks:
            if k > n:
                continue
            for solver, cutoff, fit_key in [
                ("pot",     POT_MAX_N,     "pot_sparse"),
                ("ortools", ORTOOLS_MAX_N, "ortools_sparse"),
            ]:
                if (solver, n, k) in measured_sparse:
                    continue
                if fit_key not in fits:
                    continue
                f = fits[fit_key]
                t_hat = f["a"] * n ** f["b"] * k ** f["c"]
                extra.append({
                    "scenario": "sparse_cold", "n": n, "k": k,
                    "solver": solver, "warm_ratio": None,
                    "wall_s": t_hat, "peak_mb": None,
                    "cost": None, "marginal_err_a": None, "marginal_err_b": None,
                    "extrapolated": True,
                })

    return cells + extra
```

- [ ] **Step 4: Wire `_compute_fits` and `_add_extrapolated_cells` into `main()`**

In `main()`, change:

```python
    out = {"meta": _meta(tag), "cells": cells, "fits": {}}
```

to:

```python
    fits = _compute_fits(cells)
    cells = _add_extrapolated_cells(cells, fits, dense_ns, knn_ns, knn_ks)
    out = {"meta": _meta(tag), "cells": cells, "fits": fits}
```

- [ ] **Step 5: Re-run bench.py --quick to verify fits key is populated (or empty for quick with few data points)**

```bash
python benchmarks/bench.py --quick 2>&1 | tail -3
python -c "
import json
d = json.loads(open('benchmarks/results/bench_quick.json').read())
print('fits:', d['fits'])
extrap = [c for c in d['cells'] if c['extrapolated']]
print('extrapolated cells:', len(extrap))
"
```

Expected: `fits` is either `{}` (if pot/ortools have fewer than 3 measured points at `--quick`) or populated with fit dicts. `extrapolated cells` may be 0 for `--quick` — this is fine; extrapolation is most meaningful at `--mid` and full.

- [ ] **Step 6: Run full tests**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/bench.py tests/test_benchmarks.py
git commit -m "feat(bench): power-law extrapolation — fits key + extrapolated cells"
```

---

## Task 4: `report.py` — 4 PNG figures

**Files:**
- Create: `benchmarks/report.py`

### Background

`report.py` reads `benchmarks/results/bench_{tag}.json` and produces 4 PNGs in `benchmarks/results/figures/`:

| Figure | Type | Content |
|---|---|---|
| `dense_cold.png` | Line plot (log-log) | wall_s vs n: sparse_ot (solid), pot (solid→dashed), ortools (solid→dashed) |
| `sparse_cold.png` | Heatmap | speedup ratio sparse_ot/pot at each (n, k); dashed contour at 1× |
| `warm_speedup.png` | Line plot (log-log) | cold vs warm wall_s vs n at warm_ratio=0.25 (fixed k=knn_ks[0]) |
| `accuracy.png` | Scatter | \|cost_sparse_ot − cost_pot\| / cost_pot per measured cell |

CLI:
```bash
python benchmarks/report.py --quick
python benchmarks/report.py --mid
python benchmarks/report.py           # reads bench.json
python benchmarks/report.py --print-tables   # also prints Markdown to stdout
```

- [ ] **Step 1: Write an integration test for report.py**

Add to `tests/test_benchmarks.py`:

```python
@pytest.mark.timeout(600)
def test_report_quick_produces_pngs(tmp_path):
    """report.py --quick produces all 4 PNG files."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Ensure bench_quick.json exists.
    bench_out = repo_root / "benchmarks/results/bench_quick.json"
    if not bench_out.exists():
        subprocess.run(
            [sys.executable, "benchmarks/bench.py", "--quick"],
            cwd=str(repo_root), env=env, check=True, timeout=600,
        )

    result = subprocess.run(
        [sys.executable, "benchmarks/report.py", "--quick"],
        cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"report.py --quick failed:\n{result.stderr}"

    figures_dir = repo_root / "benchmarks/results/figures"
    for name in ("dense_cold.png", "sparse_cold.png", "warm_speedup.png", "accuracy.png"):
        assert (figures_dir / name).exists(), f"Missing figure: {name}"
```

- [ ] **Step 2: Run the test to confirm it fails (report.py doesn't exist)**

```bash
pytest tests/test_benchmarks.py::test_report_quick_produces_pngs -v
```

Expected: FAILED (FileNotFoundError for report.py)

- [ ] **Step 3: Create `benchmarks/report.py`**

```python
"""Report generator — reads bench_{tag}.json, writes 4 PNG figures.

Usage:
    python benchmarks/report.py --quick
    python benchmarks/report.py --mid
    python benchmarks/report.py                 # reads bench.json
    python benchmarks/report.py --print-tables  # also prints Markdown tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SOLVER_STYLE = {
    "sparse_ot": {"color": "tab:blue",   "marker": "o", "label": "sparse-ot"},
    "pot":        {"color": "tab:orange", "marker": "s", "label": "POT (ot.emd)"},
    "ortools":    {"color": "tab:green",  "marker": "^", "label": "OR-Tools"},
}


def _load(tag):
    suffix  = f"_{tag}" if tag != "full" else ""
    p = RESULTS_DIR / f"bench{suffix}.json"
    if not p.exists():
        print(f"error: {p} not found — run bench.py {('--' + tag) if tag != 'full' else ''} first",
              file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text())


# ── figure 1: dense_cold ────────────────────────────────────────────────────

def fig_dense_cold(cells, figures_dir):
    dense = [c for c in cells if c["scenario"] == "dense_cold"]
    if not dense:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, style in SOLVER_STYLE.items():
        measured  = sorted(
            (c["n"], c["wall_s"]) for c in dense
            if c["solver"] == solver and not c["extrapolated"]
        )
        extrap    = sorted(
            (c["n"], c["wall_s"]) for c in dense
            if c["solver"] == solver and c["extrapolated"]
        )
        if measured:
            xs, ys = zip(*measured)
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    label=style["label"])
        if extrap:
            xs, ys = zip(*extrap)
            ax.plot(xs, ys, color=style["color"], linestyle="--", marker=style["marker"],
                    alpha=0.6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel("wall time (s)")
    ax.set_title("Dense cold-start: wall time vs n")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "dense_cold.png", dpi=150)
    plt.close(fig)


# ── figure 2: sparse_cold heatmap ───────────────────────────────────────────

def fig_sparse_cold(cells, figures_dir):
    # Only use measured (non-extrapolated) cells where both sparse_ot and pot ran.
    measured = [c for c in cells if c["scenario"] == "sparse_cold" and not c["extrapolated"]]

    # Build (n, k) sets for each solver.
    ot_map  = {(c["n"], c["k"]): c["wall_s"] for c in measured if c["solver"] == "sparse_ot"}
    pot_map = {(c["n"], c["k"]): c["wall_s"] for c in measured if c["solver"] == "pot"}

    common = sorted(set(ot_map) & set(pot_map))
    if not common:
        # Nothing to plot — write a placeholder.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No overlapping (n, k) for sparse_ot and pot",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(figures_dir / "sparse_cold.png", dpi=150)
        plt.close(fig)
        return

    ns = sorted({p[0] for p in common})
    ks = sorted({p[1] for p in common})
    grid = np.full((len(ns), len(ks)), np.nan)
    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            if (n, k) in ot_map and (n, k) in pot_map:
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid[i, j] = np.log10(ot_map[(n, k)] / pot_map[(n, k)])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("k (edges per source)")
    ax.set_ylabel("n (problem size)")
    ax.set_title("Sparse cold: log10(sparse-ot / POT) wall time (blue = sparse-ot faster)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(sparse-ot / POT)")
    if np.any(np.isfinite(grid)):
        try:
            ax.contour(grid, levels=[0.0], colors="black", linewidths=1.5)
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(figures_dir / "sparse_cold.png", dpi=150)
    plt.close(fig)


# ── figure 3: warm_speedup ──────────────────────────────────────────────────

def fig_warm_speedup(cells, figures_dir):
    # Compare cold (sparse_cold, solver=sparse_ot) vs warm at warm_ratio=0.25.
    cold_map = {
        (c["n"], c["k"]): c["wall_s"]
        for c in cells
        if c["scenario"] == "sparse_cold" and c["solver"] == "sparse_ot" and not c["extrapolated"]
    }
    warm_map = {
        (c["n"], c["k"]): c["wall_s"]
        for c in cells
        if c["scenario"] == "sparse_warm" and c["solver"] == "sparse_ot"
        and c["warm_ratio"] == 0.25 and not c["extrapolated"]
    }

    # Pick a k that appears in both maps.
    common_ks = {p[1] for p in cold_map} & {p[1] for p in warm_map}
    if not common_ks:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No warm_ratio=0.25 cells to plot",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(figures_dir / "warm_speedup.png", dpi=150)
        plt.close(fig)
        return

    k_plot = min(common_ks)
    fig, ax = plt.subplots(figsize=(7, 5))

    cold_pts = sorted((n, t) for (n, k), t in cold_map.items() if k == k_plot)
    warm_pts = sorted((n, t) for (n, k), t in warm_map.items() if k == k_plot)

    if cold_pts:
        xs, ys = zip(*cold_pts)
        ax.plot(xs, ys, color="tab:blue", marker="o", label="cold")
    if warm_pts:
        xs, ys = zip(*warm_pts)
        ax.plot(xs, ys, color="tab:orange", marker="s", label=f"warm (ratio=0.25, k={k_plot})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel("wall time (s)")
    ax.set_title(f"Warm-start speedup at k={k_plot} (warm_ratio=0.25)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "warm_speedup.png", dpi=150)
    plt.close(fig)


# ── figure 4: accuracy ──────────────────────────────────────────────────────

def fig_accuracy(cells, figures_dir):
    # Build (scenario, n, k) → cost for sparse_ot and pot.
    ot_cost  = {
        (c["scenario"], c["n"], c["k"]): c["cost"]
        for c in cells if c["solver"] == "sparse_ot" and not c["extrapolated"] and c["cost"] is not None
    }
    pot_cost = {
        (c["scenario"], c["n"], c["k"]): c["cost"]
        for c in cells if c["solver"] == "pot" and not c["extrapolated"] and c["cost"] is not None
    }

    xs, ys, colors = [], [], []
    scenario_color = {"dense_cold": "tab:blue", "sparse_cold": "tab:orange"}
    for key in sorted(set(ot_cost) & set(pot_cost)):
        co = ot_cost[key]; cp = pot_cost[key]
        rel_err = abs(co - cp) / max(abs(cp), 1e-30)
        xs.append(key[1])         # n
        ys.append(rel_err)
        colors.append(scenario_color.get(key[0], "gray"))

    fig, ax = plt.subplots(figsize=(7, 5))
    if xs:
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=40)
        ax.axhline(1e-10, color="k", linestyle="--", linewidth=0.8, label="1e-10")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n"); ax.set_ylabel("|cost_sparse_ot − cost_pot| / cost_pot")
        ax.set_title("Correctness: sparse-ot vs POT (blue=dense, orange=sparse)")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No overlapping sparse_ot + pot cells",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy.png", dpi=150)
    plt.close(fig)


# ── Markdown tables ──────────────────────────────────────────────────────────

def _fmt_time(t):
    if t is None:
        return "—"
    if t < 0.001:
        return f"{t*1000:.2f}ms"
    if t < 1:
        return f"{t:.3f}s"
    if t < 60:
        return f"{t:.1f}s"
    return f"~{t/60:.0f}min"


def print_tables(cells):
    print("\n### Dense cold\n")
    dense = sorted(
        [c for c in cells if c["scenario"] == "dense_cold"],
        key=lambda c: (c["n"], c["solver"])
    )
    print("| n | sparse_ot | POT | OR-Tools |")
    print("|--:|----------:|----:|---------:|")
    ns_done = set()
    by_n = {}
    for c in dense:
        by_n.setdefault(c["n"], {})[c["solver"]] = c
    for n in sorted(by_n):
        row = by_n[n]
        sot = _fmt_time(row.get("sparse_ot", {}).get("wall_s"))
        pot = _fmt_time(row.get("pot", {}).get("wall_s"))
        ort = _fmt_time(row.get("ortools", {}).get("wall_s"))
        extrap_flag = "~" if any(row.get(s, {}).get("extrapolated", False) for s in ("pot", "ortools")) else ""
        print(f"| {n:,} | {sot} | {extrap_flag}{pot} | {extrap_flag}{ort} |")

    print("\n### Warm-start speedup (warm_ratio=0.25)\n")
    print("| n | k | cold | warm | speedup |")
    print("|--:|--:|-----:|-----:|--------:|")
    cold_map = {(c["n"], c["k"]): c["wall_s"]
                for c in cells if c["scenario"] == "sparse_cold" and c["solver"] == "sparse_ot" and not c["extrapolated"]}
    warm_map = {(c["n"], c["k"]): c["wall_s"]
                for c in cells if c["scenario"] == "sparse_warm" and c["solver"] == "sparse_ot"
                and c["warm_ratio"] == 0.25 and not c["extrapolated"]}
    for (n, k) in sorted(set(cold_map) & set(warm_map)):
        cold_s = cold_map[(n, k)]; warm_s = warm_map[(n, k)]
        speedup = cold_s / warm_s if warm_s > 0 else float("inf")
        print(f"| {n:,} | {k} | {_fmt_time(cold_s)} | {_fmt_time(warm_s)} | {speedup:.1f}× |")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mid",   action="store_true")
    ap.add_argument("--print-tables", action="store_true")
    args = ap.parse_args()

    tag = "quick" if args.quick else ("mid" if args.mid else "full")
    data = _load(tag)
    cells = data["cells"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_dense_cold(cells, FIGURES_DIR)
    fig_sparse_cold(cells, FIGURES_DIR)
    fig_warm_speedup(cells, FIGURES_DIR)
    fig_accuracy(cells, FIGURES_DIR)
    print(f"figures in: {FIGURES_DIR}")

    if args.print_tables:
        print_tables(cells)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the report integration test**

```bash
pytest tests/test_benchmarks.py::test_report_quick_produces_pngs -v
```

Expected: PASSED — 4 PNG files appear under `benchmarks/results/figures/`.

- [ ] **Step 5: Visually sanity-check one figure**

```bash
open benchmarks/results/figures/dense_cold.png 2>/dev/null || \
  python -c "from PIL import Image; Image.open('benchmarks/results/figures/dense_cold.png').show()" 2>/dev/null || \
  echo "Check benchmarks/results/figures/dense_cold.png manually"
```

Confirm: log-log axes, at least one line, no empty plot.

- [ ] **Step 6: Run full tests**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/report.py tests/test_benchmarks.py
git commit -m "feat(bench): report.py — 4 PNG figures (dense_cold, sparse_cold, warm_speedup, accuracy)"
```

---

## Task 5: Cleanup — delete legacy files and update README

**Files:**
- Delete: `benchmarks/bench_solvers.py`
- Delete: `benchmarks/bench_refine.py`
- Delete: `benchmarks/generate_report.py`
- Modify: `README.md` (replace Benchmarks + Benchmark results sections)

### Background

The spec says these files are deleted. The README benchmark section is rewritten to ~350 lines total with 4 blocks embedding the 4 PNGs. Other README sections (quickstart, feasibility, convergence, memory cutoffs, releasing, warm-starting) are preserved.

The current README has "Benchmarks" (lines ~105–131) and "Benchmark results" (lines ~133–229) that reference the old scripts and tables. These are replaced with the new structure.

- [ ] **Step 1: Delete the legacy benchmark scripts**

```bash
git rm benchmarks/bench_solvers.py benchmarks/bench_refine.py benchmarks/generate_report.py
```

- [ ] **Step 2: Run the full test suite immediately to confirm nothing is broken**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all pass (CI doesn't reference the deleted scripts; `test_bench_quick_smoke` already uses `bench.py`).

- [ ] **Step 3: Rewrite the README benchmark sections**

In `README.md`, find the line:

```
## Benchmarks
```

(approximately line 104). Replace everything from that line through the end of the "## Accuracy" subsection (approximately line 219, the line before `## Memory cutoffs`) with the following. Keep all content before and after untouched.

Replacement text (paste this block exactly):

```markdown
## Benchmarks

```bash
python benchmarks/bench.py --quick    # ~30 s (used by CI)
python benchmarks/bench.py --mid      # ~15 min
python benchmarks/bench.py            # full sweep (hours)
python benchmarks/report.py --quick   # produce figures from bench_quick.json
```

Results are written to `benchmarks/results/bench_{tag}.json` (flat `cells` list + power-law `fits`). Figures go to `benchmarks/results/figures/`.

## Benchmark results

Numbers below are from `python benchmarks/bench.py --mid` on an Apple-Silicon laptop (Sonoma, 64 GB, Apple M3 Pro). Wall times are median of 1 run. `~` marks power-law-extrapolated competitor wall times (R² ≥ 0.95 required; see `fits` in the JSON for coefficients).

### Dense cold-start

![dense cold](benchmarks/results/figures/dense_cold.png)

sparse-ot and POT share the same C++ engine (POT vendors Bonneel's network simplex). The small wrapping overhead (~15%) disappears at large n where POT's default `numItermax = 100 000` truncates before convergence while our problem-size-aware default does not.

### Sparse cold-start

![sparse cold](benchmarks/results/figures/sparse_cold.png)

kNN-grid CSR problems. Heatmap shows log₁₀(sparse-ot / POT) wall time; blue = sparse-ot faster. POT and OR-Tools are measured only for n ≤ 2 000; dashed contour marks the 1× crossover.

At n ≥ 4 000 with moderate k (k ≤ n/20), sparse-ot wins by 5–15× on time and uses <10 MB vs the O(n²) cost matrix that a dense solver would require.

### Warm-start speedup

![warm speedup](benchmarks/results/figures/warm_speedup.png)

`warm_ratio=0.25` means the warm solve uses k_warm = k/4 edges per row; the refinement step completes on the full k-edge support. Wall time shown is the refinement step only (phase 2). The cold baseline comes from the sparse cold-start cells.

### Correctness

![accuracy](benchmarks/results/figures/accuracy.png)

All measured sparse-ot cells agree with POT to better than 1e-10 relative cost error. OR-Tools rounds costs to integers (scale factor 10⁶), so agreement with sparse-ot is bounded at ~1e-6. Marginal errors stay at machine precision (worst case 2.5 × 10⁻¹⁶) across all cells.
```

- [ ] **Step 4: Verify the README renders correctly (line count check)**

```bash
wc -l README.md
grep -n "^## " README.md
```

Expected: ~220–260 lines total; all major section headers visible.

- [ ] **Step 5: Run full tests one final time**

```bash
pytest tests/ -v -m "not slow" --timeout=300
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README benchmark section — unified bench.py commands + 4 embedded figures"
git add -A
git status
# Should show: nothing to commit (all tracked)
```

If `git status` shows untracked deletions or other changes, stage and amend or create a separate cleanup commit:

```bash
git add benchmarks/bench_solvers.py benchmarks/bench_refine.py benchmarks/generate_report.py
git commit -m "chore: delete legacy bench_solvers.py, bench_refine.py, generate_report.py"
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Unified `bench.py` orchestrator | Task 2 |
| `solvers.py` adapters with SolveResult | Task 1 |
| `report.py` with 4 figures | Task 4 |
| Delete `bench_solvers.py`, `bench_refine.py`, `generate_report.py` | Task 5 |
| `dense_cold` scenario | Task 2 |
| `sparse_cold` scenario | Task 2 |
| `sparse_warm` scenario | Task 2 |
| pot skip at n>2000 or nnz>100k | Task 1 |
| ortools skip at n>2000 or nnz>500k | Task 1 |
| SCALE=1_000_000 for ortools | Task 1 |
| Power-law fit + extrapolated cells | Task 3 |
| R²<0.95 → no fit | Task 3 |
| Output schema: flat cells + fits | Task 2 |
| `--quick`/`--mid`/full CLI flags | Task 2 |
| `bench_quick.json` correct schema | Task 2 (smoke test) |
| 4 PNGs produced | Task 4 |
| `report.py --quick` works | Task 4 |
| README benchmark section rewritten | Task 5 |
| All pytest tests pass | verified each task |
| sparse_ot vs pot cost error < 1e-10 | Task 4 (`accuracy.png`) |

**No placeholders present.**

**Type consistency:** `SolveResult` defined in Task 1, used identically in Tasks 2–4. `_cell()` helper defined in Task 2, unchanged in Task 3. `cells` list structure (dict with fixed keys) consistent across all tasks.
