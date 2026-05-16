import pytest
from sparse_ot.feasibility import InfeasibleProblemError


def test_exception_attributes():
    err = InfeasibleProblemError(
        imbalance=0.25,
        component_sources=[0, 1, 2],
        component_targets=[5, 6],
    )
    assert isinstance(err, ValueError)
    assert err.imbalance == 0.25
    assert err.component_sources == [0, 1, 2]
    assert err.component_targets == [5, 6]
    assert "0.25" in str(err)
