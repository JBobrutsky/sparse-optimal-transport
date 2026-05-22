"""Unified benchmark orchestrator: dense_cold, sparse_cold, sparse_warm_expand,
sparse_warm_perturb (plus matched sparse_cold_expand and sparse_cold_abs baselines).

Usage:
    python benchmarks/bench.py --quick   -> benchmarks/results/bench_quick.json
    python benchmarks/bench.py --mid     -> benchmarks/results/bench_mid.json
    python benchmarks/bench.py           -> benchmarks/results/bench.json
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
DENSE_NS_QUICK = [200]
DENSE_NS_MID   = [200, 500, 1_000, 2_000, 4_000]
DENSE_NS_FULL  = [200, 500, 1_000, 2_000, 4_000, 8_192]

KNN_NS_QUICK   = [200, 1_000]
KNN_KS_QUICK   = [4, 32]
KNN_NS_MID     = [200, 1_000, 4_000, 16_000]
KNN_KS_MID     = [2, 8, 32, 128, 512]
KNN_NS_FULL    = [200, 1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000, 16_000_000]
KNN_KS_FULL    = [2, 8, 32, 128, 512, 2_048]

KNN_NS_WARM_MAX = 1_000_000
WARM_RATIOS     = [0.5, 0.75, 0.9, 0.95]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(tag: str) -> dict:
    cpu = platform.processor() or platform.machine()
    return {
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "cpu": cpu,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _cell(
    scenario: str,
    n: int,
    k,
    solver: str,
    warm_ratio,
    result,
    extrapolated: bool = False,
) -> dict:
    """Convert a SolveResult to a cell dict with all 11 keys."""
    return {
        "scenario": scenario,
        "n": n,
        "k": k,
        "solver": solver,
        "warm_ratio": warm_ratio,
        "wall_s": result.wall_s,
        "peak_mb": result.peak_mb,
        "cost": result.cost,
        "marginal_err_a": result.marginal_err_a,
        "marginal_err_b": result.marginal_err_b,
        "extrapolated": extrapolated,
    }


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_dense_cold(dense_ns: list[int], runs: int) -> list[dict]:
    """Scenario 1: dense_cold — fully dense random OT problems."""
    from benchmarks.problems import generate_dense_random_problem
    from benchmarks.solvers import solve_sparse_ot, solve_pot, solve_ortools

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


def run_sparse_cold(knn_ns: list[int], knn_ks: list[int], runs: int) -> list[dict]:
    """Scenario 2: sparse_cold — k-NN grid OT problems, no warm start."""
    from benchmarks.problems import generate_knn_grid_problem
    from benchmarks.solvers import solve_sparse_ot, solve_pot, solve_ortools

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


def run_sparse_warm_expand(knn_ns: list[int], knn_ks: list[int], runs: int) -> list[dict]:
    """Scenario: phase 1 solves on k_warm-NN support (feasible by construction),
    phase 2 (timed) refines on the larger k_full-NN support."""
    from benchmarks.problems import generate_knn_grid_warm_expand
    from benchmarks.solvers import solve_sparse_ot
    from sparse_ot import emd

    cells = []
    for n in knn_ns:
        if n > KNN_NS_WARM_MAX:
            continue
        for k_full in knn_ks:
            if k_full > n:
                continue
            for warm_ratio in WARM_RATIOS:
                k_warm = max(2, int(round(k_full * warm_ratio)))
                if k_warm >= k_full:
                    continue
                a, b, M_warm, M_full, _ = generate_knn_grid_warm_expand(
                    n=n, k_warm=k_warm, k_full=k_full, seed=0,
                )
                M_full = M_full.tocsr()
                # Phase 1 (NOT timed)
                G_warm, info_warm = emd(a, b, M_warm, log=True)
                print(f"  sparse_warm_expand n={n} k_full={k_full} warm_ratio={warm_ratio}", flush=True)
                # Cold reference solve on M_full (matched baseline for warm-expand).
                for _ in range(runs):
                    gc.collect()
                    res_cold = solve_sparse_ot(a, b, M_full)
                    cells.append(_cell(
                        "sparse_cold_expand", n, k_full, "sparse_ot", warm_ratio, res_cold,
                    ))
                for _ in range(runs):
                    gc.collect()
                    res = solve_sparse_ot(a, b, M_full, warm=(G_warm, info_warm))
                    cells.append(_cell(
                        "sparse_warm_expand", n, k_full, "sparse_ot", warm_ratio, res,
                    ))
    return cells


def run_sparse_warm_perturb(knn_ns: list[int], knn_ks: list[int], runs: int) -> list[dict]:
    """Scenario: phase 1 solves with M_squared (L2^2 costs), phase 2 (timed)
    refines with M_abs (L1 costs) on the same k-NN support."""
    from benchmarks.problems import generate_knn_grid_warm_perturb
    from benchmarks.solvers import solve_sparse_ot
    from sparse_ot import emd

    cells = []
    for n in knn_ns:
        if n > KNN_NS_WARM_MAX:
            continue
        for k in knn_ks:
            if k > n:
                continue
            a, b, M_sq, M_abs, _ = generate_knn_grid_warm_perturb(n=n, k=k, seed=0)
            M_sq = M_sq.tocsr()
            M_abs = M_abs.tocsr()
            # Phase 1 (NOT timed)
            G_warm, info_warm = emd(a, b, M_sq, log=True)
            print(f"  sparse_warm_perturb n={n} k={k}", flush=True)
            # Cold reference solve on M_abs (matched baseline for the perturb cell).
            for _ in range(runs):
                gc.collect()
                res_cold = solve_sparse_ot(a, b, M_abs)
                cells.append(_cell(
                    "sparse_cold_abs", n, k, "sparse_ot", None, res_cold,
                ))
            for _ in range(runs):
                gc.collect()
                res = solve_sparse_ot(a, b, M_abs, warm=(G_warm, info_warm))
                cells.append(_cell(
                    "sparse_warm_perturb", n, k, "sparse_ot", None, res,
                ))
    return cells


# ---------------------------------------------------------------------------
# Power-law fitting and extrapolation
# ---------------------------------------------------------------------------

def _compute_fits(cells):
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

    # pot_sparse is intentionally skipped: POT converts sparse→dense before
    # solving, so the k dimension is noise. Fitting it with a 3-param formula
    # would be misleading; we just omit extrapolation for pot in sparse_cold.
    f = _fit_sparse("ortools")
    if f:
        fits["ortools_sparse"] = f

    return fits


def _add_extrapolated_cells(cells, fits, dense_ns, knn_ns, knn_ks):
    extra = []

    # Dense: add extrapolated cells for n > solver cutoff.
    measured_dense = {
        (c["solver"], c["n"])
        for c in cells
        if c["scenario"] == "dense_cold" and not c["extrapolated"]
    }
    for n in dense_ns:
        for solver, fit_key in [
            ("pot",     "pot_dense"),
            ("ortools", "ortools_dense"),
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

    # Sparse: add extrapolated cells for (n, k) combos the solver skipped.
    measured_sparse = {
        (c["solver"], c["n"], c["k"])
        for c in cells
        if c["scenario"] == "sparse_cold" and not c["extrapolated"]
    }
    for n in knn_ns:
        for k in knn_ks:
            if k > n:
                continue
            for solver, fit_key in [
                ("ortools", "ortools_sparse"),
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark orchestrator for sparse-optimal-transport.")
    ap.add_argument("--quick", action="store_true", help="Quick sweep (small sizes, 1 run)")
    ap.add_argument("--mid",   action="store_true", help="Mid sweep (medium sizes, 1 run)")
    args = ap.parse_args()

    if args.quick:
        dense_ns, knn_ns, knn_ks, runs, tag = (DENSE_NS_QUICK, KNN_NS_QUICK, KNN_KS_QUICK, 1, "quick")
    elif args.mid:
        dense_ns, knn_ns, knn_ks, runs, tag = (DENSE_NS_MID, KNN_NS_MID, KNN_KS_MID, 1, "mid")
    else:
        dense_ns, knn_ns, knn_ks, runs, tag = (DENSE_NS_FULL, KNN_NS_FULL, KNN_KS_FULL, 5, "full")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix   = f"_{tag}" if tag != "full" else ""
    out_path = RESULTS_DIR / f"bench{suffix}.json"

    cells = []
    cells += run_dense_cold(dense_ns, runs)
    cells += run_sparse_cold(knn_ns, knn_ks, runs)
    cells += run_sparse_warm_expand(knn_ns, knn_ks, runs)
    cells += run_sparse_warm_perturb(knn_ns, knn_ks, runs)

    fits = _compute_fits(cells)
    cells = _add_extrapolated_cells(cells, fits, dense_ns, knn_ns, knn_ks)
    out = {"meta": _meta(tag), "cells": cells, "fits": fits}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[bench] wrote {out_path} ({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
