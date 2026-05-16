# benchmarks/bench_solvers.py
"""sparse-ot benchmark driver.

Usage:
    python benchmarks/bench_solvers.py            # full sweep (hours)
    python benchmarks/bench_solvers.py --quick    # tiny sweep (seconds, for CI)
    python benchmarks/bench_solvers.py --efficiency-only
    python benchmarks/bench_solvers.py --accuracy-only

Writes:
    benchmarks/results/efficiency.json     (or _quick.json)
    benchmarks/results/accuracy.json       (or _quick.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.sparse

# Ensure repo root is importable when run as `python benchmarks/bench_solvers.py`.
import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd

# --- Memory cutoffs (16 GB machine defaults) ---
MAX_DENSE_N      = 8_192
MAX_SPARSE_NNZ   = 200_000_000
MAX_ORTOOLS_NNZ  = 500_000_000

# --- LEMON hard timeout (seconds). LEMON can hang in native code; we run it
# in a subprocess and kill if it doesn't return in time.
LEMON_TIMEOUT_S = 30.0

FULL_N = [1_000, 4_000, 8_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000, 16_000_000]
FULL_K = [2, 8, 32, 128, 512, 2048]

QUICK_N = [200, 1_000]
QUICK_K = [4, 32]

SOLVERS = ['bonneel', 'lemon', 'ortools', 'pot_reference']

RESULTS_DIR = Path(__file__).parent / "results"


def _solver_skipped(solver: str, n: int, nnz: int) -> bool:
    if solver in ('bonneel', 'pot_reference'):
        return n > MAX_DENSE_N
    if solver == 'lemon':
        return nnz > MAX_SPARSE_NNZ
    if solver == 'ortools':
        return nnz > MAX_ORTOOLS_NNZ
    raise ValueError(f"unknown solver: {solver}")


def _lemon_call_worker(a, b, M_data, M_indices, M_indptr, shape, out_q):
    """Run LEMON in a subprocess. Pickle a reconstructed CSR rather than
    passing the sparse matrix directly (cheaper to send)."""
    try:
        import scipy.sparse
        from sparse_ot import emd2
        M = scipy.sparse.csr_matrix((M_data, M_indices, M_indptr), shape=shape)
        t0 = time.perf_counter()
        _ = emd2(a, b, M, solver='lemon')
        out_q.put(("ok", time.perf_counter() - t0))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))


def _lemon_plan_worker(a, b, M_data, M_indices, M_indptr, shape, out_q):
    """Run LEMON and return the transport plan as CSR triplets."""
    try:
        import scipy.sparse
        from sparse_ot import emd
        M = scipy.sparse.csr_matrix((M_data, M_indices, M_indptr), shape=shape)
        G = emd(a, b, M, solver='lemon')
        if scipy.sparse.issparse(G):
            G_csr = G.tocsr()
            out_q.put(("ok_sparse", G_csr.data, G_csr.indices, G_csr.indptr, G_csr.shape))
        else:
            # Dense numpy result; convert to sparse triplets for transport.
            G_csr = scipy.sparse.csr_matrix(G)
            out_q.put(("ok_dense", G_csr.data, G_csr.indices, G_csr.indptr, G_csr.shape))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))


def _run_lemon_for_plan(a, b, M):
    """Run LEMON in a subprocess with a hard timeout; return (G_csr, None)
    on success or (None, error_str) on failure/timeout."""
    csr = M.tocsr()
    args = (a, b, csr.data, csr.indices, csr.indptr, csr.shape)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_lemon_plan_worker, args=args + (q,))
    p.start()
    p.join(LEMON_TIMEOUT_S)
    if p.is_alive():
        p.terminate(); p.join(2.0)
        if p.is_alive():
            p.kill(); p.join()
        return None, f"LEMON hung > {LEMON_TIMEOUT_S}s"
    try:
        result = q.get(timeout=1.0)
    except Exception:
        return None, "LEMON subprocess produced no output"
    tag = result[0]
    if tag == "err":
        return None, result[1]
    _, data, indices, indptr, shape = result
    G = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    return G, None


def _time_lemon(a, b, M, n_runs: int) -> dict:
    """Time LEMON with a subprocess hard-timeout per run. Returns a dict
    with either wall_time_s + n_runs, or error."""
    times = []
    csr = M.tocsr()
    args = (a, b, csr.data, csr.indices, csr.indptr, csr.shape)
    ctx = mp.get_context("spawn")
    for _ in range(n_runs):
        q = ctx.Queue()
        p = ctx.Process(target=_lemon_call_worker, args=args + (q,))
        p.start()
        p.join(LEMON_TIMEOUT_S)
        if p.is_alive():
            p.terminate(); p.join(2.0)
            if p.is_alive():
                p.kill(); p.join()
            return {"error": f"LEMON hung > {LEMON_TIMEOUT_S}s", "wall_time_s": None}
        try:
            tag, val = q.get(timeout=1.0)
        except Exception:
            return {"error": "LEMON subprocess produced no output", "wall_time_s": None}
        if tag == "err":
            return {"error": val, "wall_time_s": None}
        times.append(val)
    return {"wall_time_s": float(np.median(times)), "n_runs": n_runs}


def _time_solver(solver: str, a, b, M, n_runs: int) -> dict:
    """Return wall_time_median_s and peak_memory_mb for `solver` on (a, b, M).
    Bonneel/OR-Tools/POT run in-process; LEMON runs in a subprocess with
    hard timeout. Memory is measured in-process only for the in-process path."""
    if solver == 'lemon':
        res = _time_lemon(a, b, M, n_runs)
        # No in-process peak memory measurement when we ran in a subprocess.
        res.setdefault("peak_memory_mb", None)
        return res

    import ot
    from sparse_ot import emd2
    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        if solver == 'pot_reference':
            M_dense = M.toarray() if scipy.sparse.issparse(M) else np.asarray(M)
            _ = ot.emd2(a, b, M_dense)
        else:
            _ = emd2(a, b, M, solver=solver)
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_time_s": float(np.median(times)),
        "peak_memory_mb": peak_bytes / (1024 ** 2),
        "n_runs": n_runs,
    }


def _k_grid(n: int, base_ks: list[int]) -> list[int]:
    ks = set(base_ks)
    ks.add(max(2, n // 10))
    ks.add(n)
    return sorted(k for k in ks if k <= n)


def run_efficiency_sweep(ns: list[int], base_ks: list[int], n_runs: int) -> dict:
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks):
            results[str(n)][str(k)] = {}
            try:
                a, b, M = generate_knn_grid_problem(n=n, k=k, seed=0)
                nnz = M.nnz
            except MemoryError:
                for s in SOLVERS:
                    results[str(n)][str(k)][s] = None
                continue
            for solver in SOLVERS:
                if _solver_skipped(solver, n, nnz):
                    results[str(n)][str(k)][solver] = None
                    continue
                try:
                    res = _time_solver(solver, a, b, M, n_runs=n_runs)
                    res["nnz"] = int(nnz)
                    results[str(n)][str(k)][solver] = res
                except Exception as e:
                    results[str(n)][str(k)][solver] = {
                        "error": f"{type(e).__name__}: {e}",
                        "nnz": int(nnz),
                    }
                print(f"  n={n} k={k} solver={solver} -> "
                      f"{results[str(n)][str(k)][solver]}", flush=True)
    return results


def _accuracy_for_cell(solver: str, a, b, M, cost_ref):
    """Compute (cost, rel_cost_err, feas_a, feas_b) for one solver on (a, b, M).

    LEMON is wrapped in a subprocess hard-timeout (returns dict with 'error').
    """
    if solver == 'lemon':
        G, err = _run_lemon_for_plan(a, b, M)
        if err is not None:
            return {"error": err}
    else:
        try:
            G = emd(a, b, M, solver=solver)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    if scipy.sparse.issparse(G):
        cost = float(G.multiply(M).sum())
        row_sum = np.asarray(G.sum(axis=1)).ravel()
        col_sum = np.asarray(G.sum(axis=0)).ravel()
    else:
        M_dense = M.toarray() if scipy.sparse.issparse(M) else np.asarray(M)
        cost = float((G * M_dense).sum())
        row_sum = G.sum(axis=1)
        col_sum = G.sum(axis=0)
    feas_a = float(np.max(np.abs(row_sum - a)))
    feas_b = float(np.max(np.abs(col_sum - b)))
    rel = None if cost_ref is None else abs(cost - cost_ref) / max(abs(cost_ref), 1e-15)
    return {
        "cost": cost,
        "cost_ref": cost_ref,
        "rel_cost_err": rel,
        "feasibility_a": feas_a,
        "feasibility_b": feas_b,
    }


def run_accuracy_sweep(ns: list[int], base_ks: list[int]) -> dict:
    import ot
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks):
            results[str(n)][str(k)] = {}
            a, b, M = generate_knn_grid_problem(n=n, k=k, seed=0)
            nnz = M.nnz
            cost_ref = None
            if n <= 64_000 and n <= MAX_DENSE_N:
                cost_ref = float(ot.emd2(a, b, M.toarray()))
            for solver in ('lemon', 'ortools'):
                if _solver_skipped(solver, n, nnz):
                    results[str(n)][str(k)][solver] = None
                    continue
                results[str(n)][str(k)][solver] = _accuracy_for_cell(solver, a, b, M, cost_ref)
                print(f"  acc n={n} k={k} solver={solver} -> "
                      f"{results[str(n)][str(k)][solver]}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--efficiency-only", action="store_true")
    ap.add_argument("--accuracy-only",   action="store_true")
    args = ap.parse_args()

    if args.quick:
        ns, base_ks, n_runs = QUICK_N, QUICK_K, 1
    else:
        ns, base_ks, n_runs = FULL_N, FULL_K, 5

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.accuracy_only:
        print(f"[efficiency] ns={ns} ks={base_ks} n_runs={n_runs}", flush=True)
        eff = run_efficiency_sweep(ns, base_ks, n_runs)
        out = RESULTS_DIR / ("efficiency_quick.json" if args.quick else "efficiency.json")
        out.write_text(json.dumps(eff, indent=2))
        print(f"[efficiency] wrote {out}", flush=True)

    if not args.efficiency_only:
        print(f"[accuracy] ns={ns} ks={base_ks}", flush=True)
        acc = run_accuracy_sweep(ns, base_ks)
        out = RESULTS_DIR / ("accuracy_quick.json" if args.quick else "accuracy.json")
        out.write_text(json.dumps(acc, indent=2))
        print(f"[accuracy] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
