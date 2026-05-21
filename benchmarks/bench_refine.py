"""sparse-ot warm-start refinement benchmark.

Sweep (n, k_full, k_warm_ratio) on the seeded knn-grid problem from
benchmarks/problems.py. For each cell, time three paths:

  cold_full         : sparse_ot.emd(a, b, M_full) from scratch.
  refine            : sparse_ot.emd(a, b, M_warm, log=True)
                      + sparse_ot.emd(a, b, M_full, warm_start=...);
                      reported as the combined wall time (user-visible cost).
  refine_amortized  : refinement step only (cold sub-solve is "free" if
                      already paid for some other reason).

Each cell asserts cost equality between cold_full and refine within 1e-6
relative.

Usage:
    python benchmarks/bench_refine.py            # full sweep
    python benchmarks/bench_refine.py --quick    # tiny sweep (seconds)

Writes benchmarks/results/refine{_quick,}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd

RESULTS_DIR = Path(__file__).parent / "results"

NS_QUICK = [200, 1_000]
KS_FULL_QUICK = [16, 64]
WARM_RATIOS_QUICK = [0.25, 1.0]

NS_FULL = [1_000, 10_000, 100_000]
KS_FULL_FULL = [32, 128, 512]
WARM_RATIOS_FULL = [0.1, 0.25, 0.5, 1.0]


def _time_once(call):
    t0 = time.perf_counter()
    out = call()
    t1 = time.perf_counter()
    return t1 - t0, out


def _restrict_to_k(M_full_csr, k_warm):
    """Subselect the k_warm cheapest edges per row from M_full's support.

    Returns a CSR with the same shape; row i keeps min(k_warm, row_nnz)
    edges.
    """
    n, m = M_full_csr.shape
    indptr = M_full_csr.indptr
    indices = M_full_csr.indices
    data = M_full_csr.data

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
        shape=(n, m),
    )


def _run_cell(n, k_full, warm_ratio, seed=0):
    a, b, M_full, _w = generate_knn_grid_problem(n, k_full, seed=seed)
    k_warm = max(2, int(round(k_full * warm_ratio)))
    M_warm = _restrict_to_k(M_full, k_warm)

    t_cold, (G_cold, info_cold_full) = _time_once(
        lambda: emd(a, b, M_full, log=True)
    )

    t_phase1, (G_warm, info_warm) = _time_once(
        lambda: emd(a, b, M_warm, log=True)
    )
    t_phase2, (G_refined, info_refined) = _time_once(
        lambda: emd(a, b, M_full, warm_start=(G_warm, info_warm), log=True)
    )

    # Correctness gate.
    cold_cost = info_cold_full["cost"]
    ref_cost = info_refined["cost"]
    rel = abs(ref_cost - cold_cost) / max(abs(cold_cost), 1e-30)
    if rel > 1e-6:
        raise AssertionError(
            f"refine vs cold cost mismatch: rel={rel:.3e} "
            f"(cold={cold_cost!r}, refined={ref_cost!r}) "
            f"@ n={n} k_full={k_full} warm_ratio={warm_ratio}"
        )

    return {
        "n": n,
        "k_full": k_full,
        "k_warm": k_warm,
        "warm_ratio": warm_ratio,
        "cold_full_sec": t_cold,
        "phase1_sec": t_phase1,
        "phase2_sec": t_phase2,
        "refine_sec": t_phase1 + t_phase2,
        "refine_amortized_sec": t_phase2,
        "warm_start_optimal": info_refined["refine"]["warm_start_optimal"],
        "edges_added": info_refined["refine"]["edges_added"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        ns, ks, ratios = NS_QUICK, KS_FULL_QUICK, WARM_RATIOS_QUICK
        out_name = "refine_quick.json"
    else:
        ns, ks, ratios = NS_FULL, KS_FULL_FULL, WARM_RATIOS_FULL
        out_name = "refine.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        for k in ks:
            for r in ratios:
                if k > n:
                    continue
                print(f"running n={n} k_full={k} warm_ratio={r}", flush=True)
                rows.append(_run_cell(n, k, r))

    out = RESULTS_DIR / out_name
    with out.open("w") as f:
        json.dump({"cells": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
