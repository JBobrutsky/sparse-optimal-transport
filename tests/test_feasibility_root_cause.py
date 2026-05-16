# HYPOTHESIS: random Dirichlet marginals on a narrow k-NN band graph violate Hall's condition (prefix sum(a[0..i]) > sum(b[N([0..i])])), so the bipartite OT is genuinely infeasible; LEMON/OR-Tools correctly report INFEASIBLE. Secondary: sparse_utils.to_csr() calls eliminate_zeros(), dropping the diagonal (zero-cost) edges from the band graph and shrinking each source's window to k-1 cols — which makes even some otherwise-feasible band problems infeasible.
import numpy as np
import pytest
from benchmarks.problems import generate_knn_grid_problem
from sparse_ot import emd


@pytest.mark.parametrize("n,k", [(200, 4), (200, 20), (1000, 4), (1000, 32)])
def test_band_graph_is_feasible(n, k):
    a, b, M, _ = generate_knn_grid_problem(n, k, seed=0)
    # band k-NN on the same 1D grid is connected → feasible
    G = emd(a, b, M, solver='lemon')
    assert G.sum() == pytest.approx(1.0, abs=1e-9)
