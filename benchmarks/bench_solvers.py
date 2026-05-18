# benchmarks/bench_solvers.py
"""sparse-ot Bonneel-only benchmark driver.

Configurations compared:
  bonneel_dense   – pass a dense np.ndarray to sparse_ot.emd
  bonneel_sparse  – pass a scipy.sparse.csr_matrix to sparse_ot.emd
  pot_reference   – call ot.emd on the dense representation

Usage:
    python benchmarks/bench_solvers.py            # full sweep (hours)
    python benchmarks/bench_solvers.py --quick    # tiny sweep (seconds, for CI)
    python benchmarks/bench_solvers.py --efficiency-only
    python benchmarks/bench_solvers.py --accuracy-only

Writes:
    benchmarks/results/efficiency.json     (or _quick/_mid.json)
    benchmarks/results/accuracy.json       (or _quick/_mid.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.sparse

import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd

# --- Memory cutoffs (16 GB machine defaults) ---
MAX_DENSE_N    = 8_192
MAX_SPARSE_NNZ = 200_000_000

FULL_N  = [1_000, 4_000, 8_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000, 16_000_000]
FULL_K  = [2, 8, 32, 128, 512, 2048]

QUICK_N = [200, 1_000]
QUICK_K = [4, 32]

MID_N   = [200, 1_000, 4_000, 16_000]
MID_K   = [2, 8, 32, 128, 512]

CONFIGS = ['bonneel_dense', 'bonneel_sparse', 'pot_reference']

RESULTS_DIR = Path(__file__).parent / "results"


def _config_skipped(config: str, n: int, nnz: int) -> bool:
    if config in ('bonneel_dense', 'pot_reference'):
        return n > MAX_DENSE_N
    if config == 'bonneel_sparse':
        return nnz > MAX_SPARSE_NNZ
    raise ValueError(f"unknown config: {config}")


def _time_config(config: str, a, b, M_sparse, M_dense, n_runs: int) -> dict:
    """Return timing and peak-RSS for one configuration.

    M_sparse is a csr_matrix; M_dense is a contiguous float64 ndarray.
    """
    import ot as _ot

    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        if config == 'bonneel_dense':
            emd(a, b, M_dense)
        elif config == 'bonneel_sparse':
            emd(a, b, M_sparse)
        elif config == 'pot_reference':
            _ot.emd(a, b, M_dense)
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_time_s": float(np.median(times)),
        "peak_rss_mb": peak_bytes / (1024 ** 2),
        "n_runs": n_runs,
    }


def _k_grid(n: int, base_ks: list[int], add_dense: bool = True) -> list[int]:
    ks = set(base_ks)
    ks.add(max(2, n // 10))
    if add_dense:
        ks.add(n)
    return sorted(k for k in ks if k <= n)


def _sparse_to_dense_penalized(M: scipy.sparse.csr_matrix) -> np.ndarray:
    """Dense representation with absent edges set to a large finite penalty."""
    coo = M.tocoo()
    edge_mask = np.zeros(M.shape, dtype=bool)
    edge_mask[coo.row, coo.col] = True
    real_max = float(M.data.max()) if M.nnz else 1.0
    big = real_max * (M.shape[0] * M.shape[1] + 1)
    return np.where(edge_mask, M.toarray(), big).astype(np.float64, copy=False, order='C')


def run_efficiency_sweep(ns: list[int], base_ks: list[int], n_runs: int,
                         add_dense: bool = True) -> dict:
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks, add_dense=add_dense):
            results[str(n)][str(k)] = {}
            try:
                a, b, M_sp, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
                nnz = M_sp.nnz
            except MemoryError:
                for cfg in CONFIGS:
                    results[str(n)][str(k)][cfg] = None
                continue
            M_sp = M_sp.tocsr()
            M_dn = None  # materialise lazily only when needed
            for cfg in CONFIGS:
                if _config_skipped(cfg, n, nnz):
                    results[str(n)][str(k)][cfg] = None
                    continue
                if cfg in ('bonneel_dense', 'pot_reference') and M_dn is None:
                    M_dn = _sparse_to_dense_penalized(M_sp)
                try:
                    res = _time_config(cfg, a, b, M_sp, M_dn, n_runs=n_runs)
                    res["nnz"] = int(nnz)
                    results[str(n)][str(k)][cfg] = res
                except Exception as e:
                    results[str(n)][str(k)][cfg] = {
                        "error": f"{type(e).__name__}: {e}",
                        "nnz": int(nnz),
                    }
                print(f"  n={n} k={k} cfg={cfg} -> "
                      f"{results[str(n)][str(k)][cfg]}", flush=True)
    return results


FEASIBILITY_TOL = 1e-8


def _accuracy_for_config(config: str, a, b, M_sp, M_dn) -> dict:
    """Compute cost + feasibility for one configuration."""
    try:
        import ot as _ot
        if config == 'bonneel_dense':
            G = emd(a, b, M_dn)
        elif config == 'bonneel_sparse':
            G = emd(a, b, M_sp)
        elif config == 'pot_reference':
            G = _ot.emd(a, b, M_dn)
        else:
            return {"error": f"unknown config: {config}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if scipy.sparse.issparse(G):
        cost = float(G.multiply(M_sp).sum())
        row_sum = np.asarray(G.sum(axis=1)).ravel()
        col_sum = np.asarray(G.sum(axis=0)).ravel()
    else:
        M_arr = M_dn if M_dn is not None else M_sp.toarray()
        cost = float((G * M_arr).sum())
        row_sum = G.sum(axis=1)
        col_sum = G.sum(axis=0)
    return {
        "cost": cost,
        "feasibility_a": float(np.max(np.abs(row_sum - a))),
        "feasibility_b": float(np.max(np.abs(col_sum - b))),
    }


def compute_accuracy_cell(config_results: dict) -> dict:
    """Attach cost_ref / rel_cost_err using min-cost-across-feasible-configs."""
    feasible = {
        name: r for name, r in config_results.items()
        if 'error' not in r
        and r.get('feasibility_a', float('inf')) <= FEASIBILITY_TOL
        and r.get('feasibility_b', float('inf')) <= FEASIBILITY_TOL
    }
    cost_ref = min(r['cost'] for r in feasible.values()) if feasible else None

    out = {}
    for name, r in config_results.items():
        entry = dict(r)
        if 'error' in entry:
            out[name] = entry
            continue
        if name in feasible:
            entry['cost_ref'] = cost_ref
            entry['rel_cost_err'] = (
                (entry['cost'] - cost_ref) / cost_ref if cost_ref else 0.0
            )
        else:
            entry['excluded_from_reference'] = True
            entry['cost_ref'] = cost_ref
        out[name] = entry
    return out


def run_accuracy_sweep(ns: list[int], base_ks: list[int],
                       add_dense: bool = True) -> dict:
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks, add_dense=add_dense):
            a, b, M_sp, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
            nnz = M_sp.nnz
            M_sp = M_sp.tocsr()
            M_dn = None
            raw: dict = {}
            for cfg in CONFIGS:
                if _config_skipped(cfg, n, nnz):
                    continue
                if cfg in ('bonneel_dense', 'pot_reference') and M_dn is None:
                    M_dn = _sparse_to_dense_penalized(M_sp)
                cell = _accuracy_for_config(cfg, a, b, M_sp, M_dn)
                raw[cfg] = cell
                print(f"  acc n={n} k={k} cfg={cfg} -> {cell}", flush=True)
            results[str(n)][str(k)] = compute_accuracy_cell(raw)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mid",   action="store_true",
                    help="Mid sweep: n∈{200,1K,4K,16K}, k∈{2,8,32,128,512} + n//10 + n")
    ap.add_argument("--efficiency-only", action="store_true")
    ap.add_argument("--accuracy-only",   action="store_true")
    args = ap.parse_args()

    if args.quick:
        ns, base_ks, n_runs, tag, add_dense = QUICK_N, QUICK_K, 1, "quick", True
    elif args.mid:
        ns, base_ks, n_runs, tag, add_dense = MID_N, MID_K, 1, "mid", False
    else:
        ns, base_ks, n_runs, tag, add_dense = FULL_N, FULL_K, 5, None, True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eff_name = f"efficiency_{tag}.json" if tag else "efficiency.json"
    acc_name = f"accuracy_{tag}.json"   if tag else "accuracy.json"

    if not args.accuracy_only:
        print(f"[efficiency] ns={ns} ks={base_ks} n_runs={n_runs} "
              f"add_dense={add_dense}", flush=True)
        eff = run_efficiency_sweep(ns, base_ks, n_runs, add_dense=add_dense)
        out = RESULTS_DIR / eff_name
        out.write_text(json.dumps(eff, indent=2))
        print(f"[efficiency] wrote {out}", flush=True)

    if not args.efficiency_only:
        print(f"[accuracy] ns={ns} ks={base_ks} add_dense={add_dense}", flush=True)
        acc = run_accuracy_sweep(ns, base_ks, add_dense=add_dense)
        out = RESULTS_DIR / acc_name
        out.write_text(json.dumps(acc, indent=2))
        print(f"[accuracy] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
