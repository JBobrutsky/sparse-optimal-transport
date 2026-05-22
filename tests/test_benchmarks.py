# tests/test_benchmarks.py
import numpy as np
import scipy.sparse
import pytest

from benchmarks.problems import generate_knn_grid_problem


def test_generate_shapes_and_nnz():
    a, b, M, _ = generate_knn_grid_problem(n=20, k=4, seed=0)
    assert a.shape == (20,) and b.shape == (20,)
    assert scipy.sparse.issparse(M)
    assert M.shape == (20, 20)
    assert 0 < M.nnz <= 20 * 4


def test_generate_nnz_for_interior():
    """For n >> k, nnz is close to n * k."""
    a, b, M, _ = generate_knn_grid_problem(n=1000, k=8, seed=0)
    assert M.nnz >= 1000 * 8 - 8 * 8


def test_generate_fully_dense_when_k_equals_n():
    a, b, M, _ = generate_knn_grid_problem(n=10, k=10, seed=0)
    assert M.nnz == 100


def test_generate_distributions_normalized():
    a, b, M, _ = generate_knn_grid_problem(n=50, k=4, seed=0)
    np.testing.assert_allclose(a.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(b.sum(), 1.0, atol=1e-12)
    assert (a > 0).all()
    assert (b > 0).all()


def test_generate_costs_are_squared_distance():
    """Cost at edge (i, j) equals (i - j) ** 2."""
    _, _, M, _ = generate_knn_grid_problem(n=20, k=4, seed=0)
    coo = M.tocoo()
    for i, j, c in zip(coo.row, coo.col, coo.data):
        np.testing.assert_allclose(c, (int(i) - int(j)) ** 2)


def test_generate_seed_reproducibility():
    a1, b1, M1, _ = generate_knn_grid_problem(n=50, k=4, seed=42)
    a2, b2, M2, _ = generate_knn_grid_problem(n=50, k=4, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(M1.toarray(), M2.toarray())


def test_generate_seed_differs():
    a1, _, _, _ = generate_knn_grid_problem(n=50, k=4, seed=1)
    a2, _, _, _ = generate_knn_grid_problem(n=50, k=4, seed=2)
    assert not np.array_equal(a1, a2)


@pytest.mark.timeout(600)
def test_bench_quick_smoke(tmp_path):
    """`python benchmarks/bench.py --quick` exits 0 and writes bench_quick.json."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "benchmarks/bench.py", "--quick"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"bench.py --quick failed:\n{result.stderr}"

    out_path = repo_root / "benchmarks/results/bench_quick.json"
    assert out_path.exists(), "bench_quick.json not written"

    data = json.loads(out_path.read_text())
    assert "meta" in data
    assert "cells" in data
    assert "fits" in data
    assert isinstance(data["cells"], list)
    assert len(data["cells"]) > 0

    required = {"scenario", "n", "k", "solver", "warm_ratio",
                "wall_s", "peak_mb", "cost", "marginal_err_a", "marginal_err_b",
                "extrapolated"}
    for cell in data["cells"]:
        missing = required - set(cell.keys())
        assert not missing, f"Cell missing keys {missing}: {cell}"

    scenarios = {c["scenario"] for c in data["cells"]}
    assert "dense_cold" in scenarios
    assert "sparse_cold" in scenarios
    assert "sparse_warm" in scenarios

    sparse_ot_cells = [c for c in data["cells"] if c["solver"] == "sparse_ot"]
    assert len(sparse_ot_cells) > 0

    for c in data["cells"]:
        if c["scenario"] == "dense_cold":
            assert c["k"] is None

    for c in data["cells"]:
        if c["scenario"] == "sparse_cold":
            assert c["k"] is not None


@pytest.mark.timeout(600)
def test_bench_quick_fits_structure():
    """After bench.py --quick, the fits key is valid (may be empty for quick sweep)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "benchmarks/bench.py", "--quick"],
        cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"bench.py --quick failed:\n{result.stderr}"
    out_path = repo_root / "benchmarks/results/bench_quick.json"
    data = json.loads(out_path.read_text())
    assert isinstance(data["fits"], dict)
    for key, val in data["fits"].items():
        assert "a" in val and "b" in val and "r2" in val, f"bad fit for {key}: {val}"
        if key.endswith("_sparse"):
            assert "c" in val, f"sparse fit missing 'c': {key}: {val}"
        assert val["r2"] >= 0.95, f"fit r2 below threshold: {key}: {val['r2']}"
    for c in data["cells"]:
        assert "extrapolated" in c
        assert isinstance(c["extrapolated"], bool)


@pytest.mark.timeout(600)
def test_report_quick_produces_pngs(tmp_path):
    """report.py --quick produces all 4 PNG files."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Ensure bench_quick.json exists.
    bench_out = repo_root / "benchmarks/results/bench_quick.json"
    if not bench_out.exists():
        subprocess.run(
            [sys.executable, "benchmarks/bench.py", "--quick"],
            cwd=str(repo_root), env=env, check=True, timeout=600,
        )

    result = subprocess.run(
        [sys.executable, "benchmarks/report.py", "--quick"],
        cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"report.py --quick failed:\n{result.stderr}"

    figures_dir = repo_root / "benchmarks/results/figures"
    for name in ("dense_cold.png", "sparse_cold.png", "warm_speedup.png", "accuracy.png"):
        assert (figures_dir / name).exists(), f"Missing figure: {name}"


def test_fig_warm_speedup_skips_k_with_no_reduction(tmp_path):
    """fig_warm_speedup must not plot (k, warm_ratio) where k_warm == k."""
    from benchmarks.report import fig_warm_speedup

    cells = [
        # Cold
        {"scenario": "sparse_cold", "solver": "sparse_ot", "n": 1000, "k": 2,
         "warm_ratio": None, "wall_s": 0.10, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
        {"scenario": "sparse_cold", "solver": "sparse_ot", "n": 1000, "k": 32,
         "warm_ratio": None, "wall_s": 0.50, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
        # Warm (warm_ratio=0.25 → k_warm=2 for k=2 [degenerate] and k_warm=8 for k=32)
        {"scenario": "sparse_warm", "solver": "sparse_ot", "n": 1000, "k": 2,
         "warm_ratio": 0.25, "wall_s": 0.10, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
        {"scenario": "sparse_warm", "solver": "sparse_ot", "n": 1000, "k": 32,
         "warm_ratio": 0.25, "wall_s": 0.20, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
        # Another valid candidate at same ratio (k_warm=2 for k=8, ratio 0.25)
        {"scenario": "sparse_cold", "solver": "sparse_ot", "n": 1000, "k": 8,
         "warm_ratio": None, "wall_s": 0.30, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
        {"scenario": "sparse_warm", "solver": "sparse_ot", "n": 1000, "k": 8,
         "warm_ratio": 0.25, "wall_s": 0.25, "peak_mb": 0.0, "cost": 0.0,
         "marginal_err_a": 0.0, "marginal_err_b": 0.0, "extrapolated": False},
    ]
    result = fig_warm_speedup(cells, tmp_path)
    assert result == (32, 0.25), f"expected (32, 0.25), got {result}"
    assert (tmp_path / "warm_speedup.png").exists()


@pytest.mark.parametrize("n,k", [(50, 1), (50, 2), (200, 4), (1000, 8)])
def test_generator_produces_feasible_instance(n, k):
    from sparse_ot.feasibility import check_feasibility
    a, b, M, w_plan = generate_knn_grid_problem(n, k, seed=0)
    M_csr = M.tocsr()
    for i in range(n):
        cols_i = M_csr.indices[M_csr.indptr[i]:M_csr.indptr[i + 1]]
        assert i in cols_i, f"row {i} missing self-edge"
    # Float-exact marginal balance.
    assert a.sum() == b.sum(), f"sum(a)={a.sum()!r} sum(b)={b.sum()!r}"
    # Full-support marginals.
    assert (a > 0).all() and (b > 0).all()
    # Witness plan is feasible: row sums == a, col sums == b.
    w_csr = w_plan.tocsr()
    assert np.allclose(np.asarray(w_csr.sum(axis=1)).ravel(), a, atol=1e-12)
    assert np.allclose(np.asarray(w_csr.sum(axis=0)).ravel(), b, atol=1e-12)
    # Union-find feasibility check passes.
    row_ptr = M_csr.indptr.astype(np.int32)
    col_idx = M_csr.indices.astype(np.int32)
    check_feasibility(a, b, row_ptr, col_idx)


