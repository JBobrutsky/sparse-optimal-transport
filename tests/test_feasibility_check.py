import pytest
import numpy as np
import scipy.sparse
from sparse_ot.feasibility import InfeasibleProblemError, check_feasibility


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


def _csr_indices(M):
    M = M.tocsr()
    return M.indptr.astype(np.int32), M.indices.astype(np.int32)


def test_connected_balanced_support_passes():
    n = 5
    M = scipy.sparse.eye(n, format='csr')  # self-edges only, balanced
    a = np.full(n, 1 / n)
    b = np.full(n, 1 / n)
    row_ptr, col_idx = _csr_indices(M)
    check_feasibility(a, b, row_ptr, col_idx)  # must not raise


def test_disconnected_balanced_support_passes():
    # Two components: sources {0,1}↔targets{0,1}, sources {2,3,4}↔targets{2,3,4}
    rows = [0, 0, 1, 1, 2, 3, 4]
    cols = [0, 1, 0, 1, 2, 3, 4]
    data = [1.0] * len(rows)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(5, 5))
    a = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
    b = np.array([0.15, 0.15, 0.1, 0.3, 0.3])  # per-component sums match
    row_ptr, col_idx = _csr_indices(M)
    check_feasibility(a, b, row_ptr, col_idx)


def test_disconnected_imbalanced_raises():
    # Each node is its own component (5 singletons). Source 0 has mass 0.2,
    # target 0 has mass 0.1 → component {0} is imbalanced.
    rows = [0, 1, 2, 3, 4]
    cols = [0, 1, 2, 3, 4]
    data = [1.0] * 5
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(5, 5))
    a = np.array([0.2, 0.1, 0.2, 0.2, 0.3])
    b = np.array([0.1, 0.1, 0.2, 0.3, 0.3])  # component {source 0, target 0}: a=0.2, b=0.1
    row_ptr, col_idx = _csr_indices(M)
    with pytest.raises(InfeasibleProblemError) as exc_info:
        check_feasibility(a, b, row_ptr, col_idx)
    err = exc_info.value
    # Worst-imbalanced component is the one with the largest |imbalance|.
    # Source 0 / target 0: a-b = 0.1; source 3 / target 3: a-b = -0.1 (also magnitude 0.1).
    # Either could be selected — accept either.
    assert abs(err.imbalance) == pytest.approx(0.1, abs=1e-12)


def test_check_feasibility_is_fast_at_scale():
    """At n=4000, k=128 (~512k edges), feasibility must complete in well under 100ms."""
    import time
    from benchmarks.problems import generate_knn_grid_problem
    from sparse_ot.feasibility import check_feasibility
    from sparse_ot.sparse_utils import to_csr

    a, b, M, _ = generate_knn_grid_problem(n=4000, k=128, seed=0)
    M = M.tocsr()
    row_ptr, col_idx, _, _, _, _ = to_csr(M, 0.0)

    # Warm up to avoid first-call import cost dominating the budget.
    check_feasibility(a, b, row_ptr, col_idx)

    t0 = time.perf_counter()
    check_feasibility(a, b, row_ptr, col_idx)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, (
        f"check_feasibility took {elapsed*1000:.0f}ms at n=4000/k=128; "
        f"expected < 100ms after vectorization"
    )
