# tests/test_sparse_utils.py
import numpy as np
import scipy.sparse
import pytest
from sparse_ot.sparse_utils import to_csr


def test_dense_all_edges():
    M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert n == 3 and m == 2 and nnz == 6
    assert row_ptr.dtype == np.int32
    assert col_idx.dtype == np.int32
    assert costs.dtype == np.float64
    assert len(row_ptr) == n + 1
    assert len(col_idx) == nnz
    assert len(costs) == nnz


def test_dense_threshold_removes_edges():
    # |0.0| <= 0.5 and |0.5| <= 0.5 are excluded; |1.0|, |2.0| remain
    M = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, cost_sparsity_threshold=0.5)
    assert nnz == 2
    assert set(zip(col_idx.tolist(), costs.tolist())) == {(1, 1.0), (1, 2.0)}


def test_dense_exact_zeros_excluded_by_default():
    # Exact zero at (0,0) must be excluded even with threshold=0.0
    M = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert nnz == 3
    # row_ptr: [0, 1, 3] — row 0 has 1 edge, row 1 has 2 edges
    assert row_ptr[0] == 0
    assert row_ptr[1] == 1
    assert row_ptr[2] == 3


def test_dense_fully_dense():
    M = np.ones((4, 5))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert n == 4 and m == 5 and nnz == 20


def test_dense_single_edge():
    M = np.zeros((3, 3))
    M[1, 2] = 5.0
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M)
    assert nnz == 1
    assert costs[0] == 5.0


def test_scipy_csr_roundtrip():
    data = np.array([1.0, 2.0, 3.0])
    row = np.array([0, 1, 1])
    col = np.array([0, 0, 1])
    M_sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp)
    assert n == 2 and m == 2 and nnz == 3
    assert row_ptr.dtype == np.int32
    assert col_idx.dtype == np.int32
    assert costs.dtype == np.float64


def test_scipy_coo_converted():
    data = np.array([1.0, 2.0])
    row = np.array([0, 1])
    col = np.array([1, 0])
    M_sp = scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp)
    assert nnz == 2


def test_scipy_sparse_applies_threshold():
    # threshold IS applied uniformly to both dense and sparse inputs (spec §3)
    data = np.array([0.1, 0.5, 1.0])
    row = np.array([0, 0, 1])
    col = np.array([0, 1, 0])
    M_sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M_sp, cost_sparsity_threshold=0.5)
    # threshold 0.5 drops |0.1| <= 0.5 and |0.5| <= 0.5; only 1.0 remains
    assert nnz == 1
    assert costs[0] == 1.0


def test_to_csr_preserves_explicit_zero_costs():
    # 3x3 with a real free self-edge at (0, 0).
    rows = np.array([0, 0, 1, 2], dtype=np.int32)
    cols = np.array([0, 1, 1, 2], dtype=np.int32)
    data = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, 0.0)
    assert nnz == 4
    # Edge (0, 0) must still be present.
    cols_of_0 = col_idx[row_ptr[0]:row_ptr[1]]
    assert 0 in cols_of_0


def test_to_csr_drops_positive_infinity_costs():
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([0, 1, 1], dtype=np.int32)
    data = np.array([0.5, np.inf, 0.7], dtype=np.float64)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, 0.0)
    assert nnz == 2
    assert np.isfinite(costs).all()


def test_to_csr_drops_negative_infinity_and_nan():
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([0, 1, 1], dtype=np.int32)
    data = np.array([-np.inf, np.nan, 0.7], dtype=np.float64)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(2, 2))
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, 0.0)
    assert nnz == 1
    assert costs[0] == 0.7


def test_to_csr_threshold_still_drops_small_costs():
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([0, 1, 1], dtype=np.int32)
    data = np.array([0.0, 0.05, 1.0], dtype=np.float64)
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(2, 2))
    # Threshold 0.1 drops the 0.05 edge AND the explicit 0.0 edge.
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, 0.1)
    assert nnz == 1
    assert costs[0] == 1.0
