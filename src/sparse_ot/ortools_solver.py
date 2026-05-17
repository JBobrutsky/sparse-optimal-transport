# src/sparse_ot/ortools_solver.py
"""OR-Tools min-cost-flow solver for sparse balanced OT.

OR-Tools requires int64 supplies, demands, and costs. Float64 inputs are
scaled by fixed factors, rounded, balanced, and unscaled on the way out.
"""

from __future__ import annotations

import numpy as np

_SUPPLY_SCALE = 10**9  # int64-safe; supports n up to ~1e9 with sub-ULP precision.


def _scale_to_int64(values: np.ndarray, scale: int) -> np.ndarray:
    """Multiply by `scale`, round, and cast to int64."""
    return np.rint(np.asarray(values, dtype=np.float64) * scale).astype(np.int64)


def _balance_supplies(
    a_int: np.ndarray, b_int: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Make sum(a_int) == sum(b_int) by absorbing the residual into the last entries."""
    diff = int(a_int.sum() - b_int.sum())
    if diff == 0:
        return a_int, b_int
    if diff > 0:
        a_int = a_int.copy()
        a_int[-1] -= diff
    else:
        b_int = b_int.copy()
        b_int[-1] += diff
    return a_int, b_int


def solve_ortools(
    a: np.ndarray,
    b: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    costs: np.ndarray,
    ortools_cost_scale: float = 1e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a sparse balanced OT problem with OR-Tools SimpleMinCostFlow.

    Returns
    -------
    rows : (k,) int32 — source indices for non-zero flow arcs
    cols : (k,) int32 — target indices for non-zero flow arcs
    vals : (k,) float64 — transport mass on each arc
    """
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError as e:
        raise ImportError(
            "OR-Tools solver requires the 'ortools' package. "
            "Install with: pip install sparse-ot[ortools]"
        ) from e

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    costs   = np.asarray(costs,   dtype=np.float64)

    n, m = len(a), len(b)
    nnz = len(col_idx)

    a_int = _scale_to_int64(a, _SUPPLY_SCALE)
    b_int = _scale_to_int64(b, _SUPPLY_SCALE)
    a_int, b_int = _balance_supplies(a_int, b_int)

    cost_int = _scale_to_int64(costs, int(ortools_cost_scale))

    arc_capacity = int(_SUPPLY_SCALE)

    src_nodes  = np.empty(nnz, dtype=np.int64)
    dst_nodes  = np.empty(nnz, dtype=np.int64)
    capacities = np.full(nnz, arc_capacity, dtype=np.int64)
    for i in range(n):
        s, e = int(row_ptr[i]), int(row_ptr[i + 1])
        src_nodes[s:e] = i
        dst_nodes[s:e] = n + col_idx[s:e].astype(np.int64)

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_ids = smcf.add_arcs_with_capacity_and_unit_cost(
        src_nodes, dst_nodes, capacities, cost_int
    )

    supplies = np.empty(n + m, dtype=np.int64)
    supplies[:n]   =  a_int
    supplies[n:]   = -b_int
    smcf.set_nodes_supplies(np.arange(n + m, dtype=np.int64), supplies)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(
            f"OR-Tools SimpleMinCostFlow did not converge (status={status})"
        )

    flows_int = np.asarray(smcf.flows(arc_ids), dtype=np.int64)
    nonzero = flows_int > 0
    rows_out = np.asarray(src_nodes[nonzero], dtype=np.int32)
    cols_out = np.asarray(dst_nodes[nonzero] - n, dtype=np.int32)
    vals_out = (flows_int[nonzero].astype(np.float64) / _SUPPLY_SCALE)

    return rows_out, cols_out, vals_out
