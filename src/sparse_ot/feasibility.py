from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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

    Complexity: O(nnz) via scipy.sparse.csgraph.connected_components (a C
    implementation), plus vectorized numpy aggregation of supply/demand per
    component. Replaces the previous pure-Python union-find loop.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    n = a.size
    m = b.size
    nnz = int(row_ptr[-1]) if row_ptr.size > 0 else 0

    # Build bipartite adjacency as a sparse (n+m, n+m) graph.
    # Source nodes 0..n-1, target nodes n..n+m-1; edge (i, n+j) for each
    # (i, j) in the support. connected_components(directed=False) treats it
    # as undirected.
    if nnz > 0:
        src = np.repeat(np.arange(n, dtype=np.int32), np.diff(row_ptr))
        tgt = col_idx + np.int32(n)
        data = np.ones(nnz, dtype=np.int8)
        adj = csr_matrix((data, (src, tgt)), shape=(n + m, n + m))
    else:
        # Empty graph: every node is its own component.
        adj = csr_matrix((n + m, n + m), dtype=np.int8)

    n_components, labels = connected_components(adj, directed=False)

    # Aggregate supply/demand by component label. bincount is faster than
    # np.add.at and exact for float64 sums in this size regime.
    supply_by_comp = np.bincount(labels[:n], weights=a, minlength=n_components)
    demand_by_comp = np.bincount(labels[n:n + m], weights=b, minlength=n_components)

    diff = supply_by_comp - demand_by_comp
    abs_diff = np.abs(diff)
    worst_idx = int(np.argmax(abs_diff))
    worst_imbalance = float(diff[worst_idx])

    if abs(worst_imbalance) > tol:
        bad_label = worst_idx
        bad_sources = np.where(labels[:n] == bad_label)[0]
        bad_targets = np.where(labels[n:n + m] == bad_label)[0]
        raise InfeasibleProblemError(
            imbalance=worst_imbalance,
            component_sources=bad_sources.tolist(),
            component_targets=bad_targets.tolist(),
        )
