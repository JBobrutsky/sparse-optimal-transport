# tests/test_benchmarks.py
import numpy as np
import scipy.sparse
import pytest

from benchmarks.problems import generate_knn_grid_problem


def test_generate_shapes_and_nnz():
    a, b, M = generate_knn_grid_problem(n=20, k=4, seed=0)
    assert a.shape == (20,) and b.shape == (20,)
    assert scipy.sparse.issparse(M)
    assert M.shape == (20, 20)
    assert 0 < M.nnz <= 20 * 4


def test_generate_nnz_for_interior():
    """For n >> k, nnz is close to n * k."""
    a, b, M = generate_knn_grid_problem(n=1000, k=8, seed=0)
    assert M.nnz >= 1000 * 8 - 8 * 8


def test_generate_fully_dense_when_k_equals_n():
    a, b, M = generate_knn_grid_problem(n=10, k=10, seed=0)
    assert M.nnz == 100


def test_generate_distributions_normalized():
    a, b, M = generate_knn_grid_problem(n=50, k=4, seed=0)
    np.testing.assert_allclose(a.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(b.sum(), 1.0, atol=1e-12)
    assert (a > 0).all()
    assert (b > 0).all()


def test_generate_costs_are_squared_distance():
    """Cost at edge (i, j) equals (i - j) ** 2."""
    _, _, M = generate_knn_grid_problem(n=20, k=4, seed=0)
    coo = M.tocoo()
    for i, j, c in zip(coo.row, coo.col, coo.data):
        np.testing.assert_allclose(c, (int(i) - int(j)) ** 2)


def test_generate_seed_reproducibility():
    a1, b1, M1 = generate_knn_grid_problem(n=50, k=4, seed=42)
    a2, b2, M2 = generate_knn_grid_problem(n=50, k=4, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(M1.toarray(), M2.toarray())


def test_generate_seed_differs():
    a1, _, _ = generate_knn_grid_problem(n=50, k=4, seed=1)
    a2, _, _ = generate_knn_grid_problem(n=50, k=4, seed=2)
    assert not np.array_equal(a1, a2)


@pytest.mark.timeout(600)
def test_bench_solvers_quick_smoke(tmp_path):
    """`python benchmarks/bench_solvers.py --quick` exits 0 and writes JSON."""
    import subprocess, sys, os
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "benchmarks/bench_solvers.py", "--quick"],
        cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (repo_root / "benchmarks/results/efficiency_quick.json").exists()
    assert (repo_root / "benchmarks/results/accuracy_quick.json").exists()
