# tests/test_routing.py
import pytest
from sparse_ot.routing import select_solver, _load_thresholds


# Routing thresholds are hardware-derived (see Plan 4 Task 10). Tests read
# the live thresholds and pick inputs straddling them so they stay valid
# across re-tunes; the logic under test is the dispatch, not the numbers.
def _thresh():
    t = _load_thresholds()
    return t["bonneel_lemon"], t["lemon_ortools"]


def test_high_k_routes_to_bonneel():
    bl, _ = _thresh()
    # k > bonneel_lemon → bonneel
    n = 10
    nnz = n * (bl + 1)
    assert select_solver(n=n, m=n, nnz=nnz) == 'bonneel'


def test_low_k_small_n_routes_to_lemon():
    bl, lo = _thresh()
    # k ≤ bonneel_lemon AND n ≤ lemon_ortools → lemon
    n = max(2, lo // 2)
    nnz = n * bl  # k = bl, not above
    assert select_solver(n=n, m=n, nnz=nnz) == 'lemon'


def test_large_n_routes_to_ortools():
    bl, lo = _thresh()
    # k ≤ bonneel_lemon AND n > lemon_ortools → ortools
    n = lo + 1
    nnz = n * bl
    assert select_solver(n=n, m=n, nnz=nnz) == 'ortools'


def test_boundary_n_just_below_ortools_is_lemon():
    bl, lo = _thresh()
    n = lo  # exactly at threshold → not above → lemon
    nnz = n * bl
    assert select_solver(n=n, m=n, nnz=nnz) == 'lemon'


def test_boundary_n_just_above_ortools():
    bl, lo = _thresh()
    n = lo + 1
    nnz = n * bl
    assert select_solver(n=n, m=n, nnz=nnz) == 'ortools'


def test_override_beats_routing():
    # Whatever the thresholds say, explicit solver= wins.
    assert select_solver(n=10, m=10, nnz=100, solver='bonneel') == 'bonneel'
    assert select_solver(n=10, m=10, nnz=10_000, solver='lemon') == 'lemon'
    assert select_solver(n=10, m=10, nnz=10_000, solver='ortools') == 'ortools'


def test_override_lemon_beats_bonneel_routing():
    # k=200 would normally → bonneel, but override wins
    assert select_solver(n=10, m=10, nnz=2000, solver='lemon') == 'lemon'


def test_override_ortools():
    assert select_solver(n=10, m=10, nnz=100, solver='ortools') == 'ortools'


def test_large_k_fully_dense_routes_to_bonneel():
    bl, _ = _thresh()
    n = max(2, bl + 1)
    # k = n > bonneel_lemon → bonneel
    assert select_solver(n=n, m=n, nnz=n * n) == 'bonneel'
