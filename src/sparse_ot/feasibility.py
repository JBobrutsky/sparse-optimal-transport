from __future__ import annotations


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
