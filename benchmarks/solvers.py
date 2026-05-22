"""Uniform SolveResult interface and solver adapter functions for benchmarking."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import scipy.sparse

POT_MAX_N = 2_000
POT_MAX_NNZ = 100_000
ORTOOLS_MAX_N = 2_000
ORTOOLS_MAX_NNZ = 500_000
SCALE = 1_000_000  # integer factor for OR-Tools


@dataclass
class SolveResult:
    cost: float
    wall_s: float
    peak_mb: float
    marginal_err_a: float  # max |G.sum(1) - a|
    marginal_err_b: float  # max |G.sum(0) - b|


def _measure(fn):
    """Call fn(), return (return_value, wall_s_float, peak_mb_float)."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()
    return out, wall_s, peak_bytes / 1024 ** 2


def _marginal_errors(G, a, b):
    """Return (err_a, err_b) for dense or CSR G."""
    if scipy.sparse.issparse(G):
        row_sums = np.asarray(G.sum(axis=1)).ravel()
        col_sums = np.asarray(G.sum(axis=0)).ravel()
    else:
        row_sums = G.sum(axis=1)
        col_sums = G.sum(axis=0)
    err_a = float(np.max(np.abs(row_sums - a)))
    err_b = float(np.max(np.abs(col_sums - b)))
    return err_a, err_b


def solve_sparse_ot(a, b, M, warm=None) -> SolveResult:
    """Calls sparse_ot.emd(a, b, M, warm_start=warm, log=True). Always runs."""
    from sparse_ot import emd

    (G, info), wall_s, peak_mb = _measure(lambda: emd(a, b, M, warm_start=warm, log=True))
    cost = info["cost"]
    err_a, err_b = _marginal_errors(G, a, b)
    return SolveResult(
        cost=float(cost),
        wall_s=wall_s,
        peak_mb=peak_mb,
        marginal_err_a=err_a,
        marginal_err_b=err_b,
    )


def solve_pot(a, b, M) -> SolveResult | None:
    """Calls ot.emd(a, b, M_dense). Returns None when n > POT_MAX_N or CSR nnz > POT_MAX_NNZ."""
    import ot

    if scipy.sparse.issparse(M):
        if M.nnz > POT_MAX_NNZ:
            return None
        n = M.shape[0]
        penalty = float(M.data.max()) * n * 10 if M.nnz > 0 else 1.0
        M_dense = np.full((n, n), penalty, dtype=np.float64)
        coo = M.tocoo()
        M_dense[coo.row, coo.col] = coo.data
    else:
        n = M.shape[0]
        M_dense = np.asarray(M, dtype=np.float64)

    if n > POT_MAX_N:
        return None

    G, wall_s, peak_mb = _measure(lambda: ot.emd(a, b, M_dense))
    cost = float(np.sum(G * M_dense))
    err_a, err_b = _marginal_errors(G, a, b)
    return SolveResult(
        cost=cost,
        wall_s=wall_s,
        peak_mb=peak_mb,
        marginal_err_a=err_a,
        marginal_err_b=err_b,
    )


def solve_ortools(a, b, M) -> SolveResult | None:
    """OR-Tools min_cost_flow. Costs scaled by SCALE=1_000_000. Returns None when n > ORTOOLS_MAX_N or nnz > ORTOOLS_MAX_NNZ."""
    from ortools.graph.python import min_cost_flow

    if scipy.sparse.issparse(M):
        n = M.shape[0]
        nnz = M.nnz
    else:
        n = M.shape[0]
        nnz = M.size

    if n > ORTOOLS_MAX_N or nnz > ORTOOLS_MAX_NNZ:
        return None

    supply_a = np.round(a * SCALE).astype(np.int64)
    supply_b = np.round(b * SCALE).astype(np.int64)
    # Fix rounding imbalance
    supply_b[0] += supply_a.sum() - supply_b.sum()

    smcf = min_cost_flow.SimpleMinCostFlow()

    # OR-Tools node layout: sources 0..n-1, sinks n..2n-1
    if scipy.sparse.issparse(M):
        coo = M.tocoo()
        for row, col, c in zip(coo.row, coo.col, coo.data):
            smcf.add_arc_with_capacity_and_unit_cost(
                int(row), n + int(col), SCALE, int(round(c * SCALE))
            )
    else:
        for i in range(n):
            for j in range(n):
                c = M[i, j]
                smcf.add_arc_with_capacity_and_unit_cost(
                    i, n + j, SCALE, int(round(c * SCALE))
                )

    for i in range(n):
        smcf.set_node_supply(i, int(supply_a[i]))
    for j in range(n):
        smcf.set_node_supply(n + j, -int(supply_b[j]))

    status, wall_s, peak_mb = _measure(smcf.solve)

    if status != smcf.OPTIMAL:
        return None

    cost = smcf.optimal_cost() / (SCALE * SCALE)

    # Reconstruct marginal errors
    row_flow = np.zeros(n)
    col_flow = np.zeros(n)
    for arc in range(smcf.num_arcs()):
        flow = smcf.flow(arc) / SCALE
        row_flow[smcf.tail(arc)] += flow
        col_flow[smcf.head(arc) - n] += flow

    err_a = float(np.max(np.abs(row_flow - a)))
    err_b = float(np.max(np.abs(col_flow - b)))

    return SolveResult(
        cost=cost,
        wall_s=wall_s,
        peak_mb=peak_mb,
        marginal_err_a=err_a,
        marginal_err_b=err_b,
    )
