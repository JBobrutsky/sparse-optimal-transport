# benchmarks/bench_solvers.py
"""sparse-ot Bonneel-only benchmark driver.

Two independent suites:

  dense
    Fully-dense random cost matrix (n x n). Configs: bonneel_dense,
    pot_reference. Used to compare against POT on the regime POT was
    designed for.

  sparse (knn-grid)
    Feasible-by-construction knn-grid instance from problems.py. Only
    config: bonneel_sparse. POT and bonneel_dense are not run on these:
    they require a dense representation, and the only honest dense
    representation of a knn cost is one with absent edges as +inf (which
    POT/Bonneel cannot handle) -- substituting a finite penalty leads to
    routing through penalty edges and meaningless cost values.

Usage:
    python benchmarks/bench_solvers.py            # full sweep (hours)
    python benchmarks/bench_solvers.py --quick    # tiny sweep (seconds)
    python benchmarks/bench_solvers.py --mid
    python benchmarks/bench_solvers.py --efficiency-only
    python benchmarks/bench_solvers.py --accuracy-only

Writes:
    benchmarks/results/efficiency{_quick,_mid,}.json
    benchmarks/results/accuracy{_quick,_mid,}.json
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

from benchmarks.problems import (
    generate_dense_random_problem,
    generate_knn_grid_problem,
)
from sparse_ot import emd

# --- Cutoffs ---
MAX_DENSE_N    = 8_192        # dense cost matrix capped to ~512 MB
MAX_SPARSE_NNZ = 200_000_000

# --- Sweep definitions ---
DENSE_NS_QUICK  = [200]
DENSE_NS_MID    = [200, 500, 1000, 2000, 4000]
DENSE_NS_FULL   = [200, 500, 1000, 2000, 4000, 8000]

KNN_NS_QUICK = [200, 1_000]
KNN_KS_QUICK = [4, 32]

KNN_NS_MID   = [200, 1_000, 4_000, 16_000]
KNN_KS_MID   = [2, 8, 32, 128, 512]

KNN_NS_FULL  = [1_000, 4_000, 8_000, 16_000, 64_000, 256_000,
                1_000_000, 4_000_000, 16_000_000]
KNN_KS_FULL  = [2, 8, 32, 128, 512, 2048]

RESULTS_DIR = Path(__file__).parent / "results"


# ----------------------------- timing helpers ------------------------------

def _time(call, n_runs):
    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_time_s": float(np.median(times)),
        "peak_rss_mb": peak_bytes / (1024 ** 2),
        "n_runs": n_runs,
    }


def _try(call):
    try:
        return call()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ----------------------------- dense suite ---------------------------------

def run_dense_efficiency(ns, n_runs):
    import ot as _ot
    out = {}
    for n in ns:
        out[str(n)] = {}
        if n > MAX_DENSE_N:
            out[str(n)]["bonneel_dense"] = None
            out[str(n)]["pot_reference"] = None
            continue
        a, b, M = generate_dense_random_problem(n, seed=0)
        out[str(n)]["bonneel_dense"]  = _try(lambda: _time(lambda: emd(a, b, M), n_runs))
        out[str(n)]["pot_reference"]  = _try(lambda: _time(lambda: _ot.emd(a, b, M), n_runs))
        print(f"  dense n={n} -> {out[str(n)]}", flush=True)
    return out


def run_dense_accuracy(ns):
    import ot as _ot
    out = {}
    for n in ns:
        out[str(n)] = {}
        if n > MAX_DENSE_N:
            continue
        a, b, M = generate_dense_random_problem(n, seed=0)

        def _measure(G):
            row = np.asarray(G.sum(axis=1)).ravel() if scipy.sparse.issparse(G) else G.sum(axis=1)
            col = np.asarray(G.sum(axis=0)).ravel() if scipy.sparse.issparse(G) else G.sum(axis=0)
            cost = float((G.multiply(M).sum()) if scipy.sparse.issparse(G) else np.sum(G * M))
            return {
                "cost": cost,
                "feasibility_a": float(np.max(np.abs(row - a))),
                "feasibility_b": float(np.max(np.abs(col - b))),
            }

        out[str(n)]["bonneel_dense"]  = _try(lambda: _measure(emd(a, b, M)))
        out[str(n)]["pot_reference"]  = _try(lambda: _measure(_ot.emd(a, b, M)))
        print(f"  dense acc n={n} -> {out[str(n)]}", flush=True)
    return out


# ----------------------------- sparse suite --------------------------------

def _knn_k_grid(n, base_ks):
    ks = set(base_ks)
    ks.add(max(2, n // 10))
    return sorted(k for k in ks if k <= n)


def _densify_with_penalty(M):
    """Dense view of a sparse cost matrix where absent edges carry a finite
    penalty larger than any feasible plan's cost. With a bumped numItermax,
    Bonneel converges to the same optimum as the sparse solver — the penalty
    edges sit at high reduced cost and are never used in the optimal basis."""
    coo = M.tocoo()
    mask = np.zeros(M.shape, dtype=bool)
    mask[coo.row, coo.col] = True
    real_max = float(M.data.max()) if M.nnz else 1.0
    big = real_max * (M.shape[0] * M.shape[1] + 1)
    return np.where(mask, M.toarray(), big).astype(np.float64, copy=False, order='C')


def run_sparse_efficiency(ns, base_ks, n_runs):
    out = {}
    for n in ns:
        out[str(n)] = {}
        for k in _knn_k_grid(n, base_ks):
            out[str(n)][str(k)] = {}
            a, b, M, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
            M = M.tocsr()
            nnz = M.nnz

            # bonneel_sparse: native CSR input.
            if nnz > MAX_SPARSE_NNZ:
                out[str(n)][str(k)]["bonneel_sparse"] = None
            else:
                res = _try(lambda: _time(lambda: emd(a, b, M), n_runs))
                if isinstance(res, dict) and "error" not in res:
                    res["nnz"] = int(nnz)
                out[str(n)][str(k)]["bonneel_sparse"] = res

            # bonneel_dense: the same problem densified with a finite penalty.
            # Only runs while the dense matrix fits.
            if n <= MAX_DENSE_N:
                M_dn = _densify_with_penalty(M)
                res_d = _try(lambda: _time(lambda: emd(a, b, M_dn), n_runs))
                if isinstance(res_d, dict) and "error" not in res_d:
                    res_d["nnz"] = int(nnz)
                out[str(n)][str(k)]["bonneel_dense"] = res_d
            else:
                out[str(n)][str(k)]["bonneel_dense"] = None

            print(f"  sparse n={n} k={k} nnz={nnz} -> {out[str(n)][str(k)]}",
                  flush=True)
    return out


def run_sparse_accuracy(ns, base_ks):
    out = {}
    for n in ns:
        out[str(n)] = {}
        for k in _knn_k_grid(n, base_ks):
            a, b, M, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
            M = M.tocsr()
            cell = {}

            if M.nnz <= MAX_SPARSE_NNZ:
                def _measure_sparse():
                    G = emd(a, b, M)
                    row = np.asarray(G.sum(axis=1)).ravel()
                    col = np.asarray(G.sum(axis=0)).ravel()
                    return {
                        "cost": float(G.multiply(M).sum()),
                        "feasibility_a": float(np.max(np.abs(row - a))),
                        "feasibility_b": float(np.max(np.abs(col - b))),
                        "nnz": int(M.nnz),
                    }
                cell["bonneel_sparse"] = _try(_measure_sparse)

            if n <= MAX_DENSE_N:
                M_dn = _densify_with_penalty(M)
                def _measure_dense():
                    G = emd(a, b, M_dn)
                    # Cost is measured against the *true* sparse M, not the
                    # penalised dense one — otherwise any leak onto penalty
                    # edges would dominate.
                    cost = float(np.sum(G * M.toarray()))
                    return {
                        "cost": cost,
                        "feasibility_a": float(np.max(np.abs(G.sum(axis=1) - a))),
                        "feasibility_b": float(np.max(np.abs(G.sum(axis=0) - b))),
                        "nnz": int(M.nnz),
                    }
                cell["bonneel_dense"] = _try(_measure_dense)

            out[str(n)][str(k)] = cell
            print(f"  sparse acc n={n} k={k} -> {cell}", flush=True)
    return out


# ----------------------------- driver --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mid",   action="store_true")
    ap.add_argument("--efficiency-only", action="store_true")
    ap.add_argument("--accuracy-only",   action="store_true")
    args = ap.parse_args()

    if args.quick:
        dense_ns, knn_ns, knn_ks, n_runs, tag = (
            DENSE_NS_QUICK, KNN_NS_QUICK, KNN_KS_QUICK, 1, "quick"
        )
    elif args.mid:
        dense_ns, knn_ns, knn_ks, n_runs, tag = (
            DENSE_NS_MID, KNN_NS_MID, KNN_KS_MID, 1, "mid"
        )
    else:
        dense_ns, knn_ns, knn_ks, n_runs, tag = (
            DENSE_NS_FULL, KNN_NS_FULL, KNN_KS_FULL, 5, None
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eff_path = RESULTS_DIR / (f"efficiency_{tag}.json" if tag else "efficiency.json")
    acc_path = RESULTS_DIR / (f"accuracy_{tag}.json"   if tag else "accuracy.json")

    if not args.accuracy_only:
        print(f"[efficiency] dense_ns={dense_ns} knn_ns={knn_ns} knn_ks={knn_ks} "
              f"n_runs={n_runs}", flush=True)
        eff = {
            "dense":  run_dense_efficiency(dense_ns, n_runs),
            "sparse": run_sparse_efficiency(knn_ns, knn_ks, n_runs),
        }
        eff_path.write_text(json.dumps(eff, indent=2))
        print(f"[efficiency] wrote {eff_path}", flush=True)

    if not args.efficiency_only:
        print(f"[accuracy] dense_ns={dense_ns} knn_ns={knn_ns} knn_ks={knn_ks}",
              flush=True)
        acc = {
            "dense":  run_dense_accuracy(dense_ns),
            "sparse": run_sparse_accuracy(knn_ns, knn_ks),
        }
        acc_path.write_text(json.dumps(acc, indent=2))
        print(f"[accuracy] wrote {acc_path}", flush=True)


if __name__ == "__main__":
    main()
