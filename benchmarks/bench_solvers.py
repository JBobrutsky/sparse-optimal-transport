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

MID_N = [200, 1_000, 4_000, 16_000]  # spec grid; see --mid notes in main()
MID_K = [2, 8, 32, 128, 512]

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


def _densify_with_penalty(M):
    """Materialize a sparse cost matrix as dense, with absent cells set to a
    large finite penalty so a dense solver (Bonneel, POT) cannot route mass
    through absent edges. Returns a contiguous float64 array."""
    if not scipy.sparse.issparse(M):
        return np.asarray(M, dtype=np.float64, order='C')
    coo = M.tocoo()
    edge_mask = np.zeros(M.shape, dtype=bool)
    edge_mask[coo.row, coo.col] = True
    real_max = float(M.data.max()) if M.nnz else 1.0
    big = real_max * (M.shape[0] * M.shape[1] + 1)
    return np.where(edge_mask, M.toarray(), big).astype(np.float64, copy=False, order='C')


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
    # Bonneel and POT solve on dense M; sparse inputs need absent cells
    # filled with a large penalty so they don't route through "free" zeros.
    M_call = _densify_with_penalty(M) if solver in ('bonneel', 'pot_reference') else M
    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        if solver == 'pot_reference':
            _ = ot.emd2(a, b, M_call)
        else:
            _ = emd2(a, b, M_call, solver=solver)
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_time_s": float(np.median(times)),
        "peak_memory_mb": peak_bytes / (1024 ** 2),
        "n_runs": n_runs,
    }


def _k_grid(n: int, base_ks: list[int], add_dense: bool = True) -> list[int]:
    ks = set(base_ks)
    ks.add(max(2, n // 10))
    if add_dense:
        ks.add(n)
    return sorted(k for k in ks if k <= n)


def run_efficiency_sweep(ns: list[int], base_ks: list[int], n_runs: int,
                         add_dense: bool = True) -> dict:
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks, add_dense=add_dense):
            results[str(n)][str(k)] = {}
            try:
                a, b, M, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
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
            M_call = _densify_with_penalty(M) if solver == 'bonneel' else M
            G = emd(a, b, M_call, solver=solver)
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


FEASIBILITY_TOL = 1e-8


def compute_accuracy_cell(solver_results: dict) -> dict:
    """Recompute cost_ref + rel_cost_err using min-across-feasible-solvers.

    solver_results maps solver_name -> dict with either 'error' or
    ('cost', 'feasibility_a', 'feasibility_b'). Returns a new dict of the
    same shape, with 'cost_ref' / 'rel_cost_err' attached to feasible
    entries and 'excluded_from_reference' on entries that ran but failed
    the feasibility tolerance. 'error' entries are passed through unchanged.
    """
    feasible = {}
    for name, r in solver_results.items():
        if 'error' in r:
            continue
        fa = r.get('feasibility_a', float('inf'))
        fb = r.get('feasibility_b', float('inf'))
        if fa is None or fb is None or fa > FEASIBILITY_TOL or fb > FEASIBILITY_TOL:
            continue
        feasible[name] = r

    if feasible:
        cost_ref = min(r['cost'] for r in feasible.values())
    else:
        cost_ref = None

    out = {}
    for name, r in solver_results.items():
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


def _pot_reference_cost(a, b, M_csr) -> float:
    """Run POT on the same restricted problem, encoding non-edges as a large
    finite cost so POT can't route through them.

    M_csr.toarray() encodes "no edge" as 0 — POT would route mass through
    those free entries and report a meaningless underestimate. Replacing
    non-edges with a value larger than the worst-case all-real-edges cost
    forces POT to solve the actual k-NN restricted problem.
    """
    import ot
    coo = M_csr.tocoo()
    edge_mask = np.zeros(M_csr.shape, dtype=bool)
    edge_mask[coo.row, coo.col] = True
    real_max = float(M_csr.data.max()) if M_csr.nnz else 1.0
    # Big enough that routing a single unit through one non-edge costs more
    # than the entire feasible plan; small enough to keep float64 precision.
    big = real_max * (M_csr.shape[0] * M_csr.shape[1] + 1)
    M_dense = np.where(edge_mask, M_csr.toarray(), big)
    return float(ot.emd2(a, b, M_dense))


def run_accuracy_sweep(ns: list[int], base_ks: list[int],
                       add_dense: bool = True) -> dict:
    results: dict = {}
    for n in ns:
        results[str(n)] = {}
        for k in _k_grid(n, base_ks, add_dense=add_dense):
            results[str(n)][str(k)] = {}
            a, b, M, _ = generate_knn_grid_problem(n=n, k=k, seed=0)
            nnz = M.nnz
            raw: dict = {}
            for solver in ('lemon', 'ortools'):
                if _solver_skipped(solver, n, nnz):
                    continue
                cell = _accuracy_for_cell(solver, a, b, M, cost_ref=None)
                cell.pop('cost_ref', None)
                cell.pop('rel_cost_err', None)
                raw[solver] = cell
                print(f"  acc n={n} k={k} solver={solver} -> {cell}", flush=True)
            # POT participates only at k == n (fully dense — solves the same problem).
            if k == n and n <= 64_000 and n <= MAX_DENSE_N:
                try:
                    cost_pot = _pot_reference_cost(a, b, M)
                    raw['pot_reference'] = {
                        'cost': cost_pot,
                        'feasibility_a': 0.0,
                        'feasibility_b': 0.0,
                    }
                except Exception as e:
                    raw['pot_reference'] = {'error': f"{type(e).__name__}: {e}"}
                print(f"  acc n={n} k={k} solver=pot_reference -> "
                      f"{raw['pot_reference']}", flush=True)
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
        # Mid sweep: skip k=n (dense) cells to avoid multi-minute OR-Tools runs
        # at large n. n_runs=1 for speed.
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
