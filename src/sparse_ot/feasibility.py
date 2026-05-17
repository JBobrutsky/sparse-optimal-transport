from __future__ import annotations

import numpy as np


class InfeasibleProblemError(ValueError):
    """Raised when the sparse cost matrix's support cannot transport (a, b).

    Attributes
    ----------
    imbalance : float
        Signed mass imbalance of the worst-violating component
        (sum(a in component) - sum(b in component)).
    component_sources : list[int]
        Source-node indices in the violating component (truncated to 10).
    component_targets : list[int]
        Target-node indices in the violating component (truncated to 10).
    """

    def __init__(self, imbalance, component_sources, component_targets):
        self.imbalance = float(imbalance)
        self.component_sources = list(component_sources[:10])
        self.component_targets = list(component_targets[:10])
        msg = (
            f"sparse cost matrix support is infeasible for the given marginals: "
            f"component imbalance sum(a) - sum(b) = {self.imbalance!r}; "
            f"sources={self.component_sources!r}, "
            f"targets={self.component_targets!r}"
        )
        super().__init__(msg)


def check_feasibility(a, b, row_ptr, col_idx, tol: float = 1e-12) -> None:
    """Raise InfeasibleProblemError if the bipartite support cannot route (a, b).

    Necessary-and-sufficient condition when sum(a) == sum(b) and edges have
    unbounded capacity: every connected component of the bipartite support
    graph must have sum(a over its sources) == sum(b over its targets).

    Complexity: O(nnz · α(n+m)) via union-find.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    n = a.size
    m = b.size

    # Union-find over n + m nodes: sources [0..n), targets [n..n+m).
    parent = np.arange(n + m, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            j = int(col_idx[p])
            union(i, n + j)

    comp_supply: dict[int, float] = {}
    comp_demand: dict[int, float] = {}
    comp_sources: dict[int, list[int]] = {}
    comp_targets: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        comp_supply[r] = comp_supply.get(r, 0.0) + float(a[i])
        comp_sources.setdefault(r, []).append(i)
    for j in range(m):
        r = find(n + j)
        comp_demand[r] = comp_demand.get(r, 0.0) + float(b[j])
        comp_targets.setdefault(r, []).append(j)

    worst_root = None
    worst_imbalance = 0.0
    for r in set(comp_supply) | set(comp_demand):
        s = comp_supply.get(r, 0.0)
        d = comp_demand.get(r, 0.0)
        diff = s - d
        if abs(diff) > abs(worst_imbalance):
            worst_imbalance = diff
            worst_root = r

    if worst_root is not None and abs(worst_imbalance) > tol:
        raise InfeasibleProblemError(
            imbalance=worst_imbalance,
            component_sources=comp_sources.get(worst_root, []),
            component_targets=comp_targets.get(worst_root, []),
        )
