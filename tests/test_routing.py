# tests/test_routing.py
import pytest
from sparse_ot.routing import select_solver


def test_high_k_routes_to_bonneel():
    # k = nnz/n = 2000/10 = 200 > 128 → bonneel
    assert select_solver(n=10, m=10, nnz=2000) == 'bonneel'


def test_low_k_small_n_routes_to_lemon():
    # k = 100/10 = 10 < 128, n=10 < 1_000_000 → lemon
    assert select_solver(n=10, m=10, nnz=100) == 'lemon'


def test_large_n_routes_to_ortools():
    # k = 4_000_000/2_000_000 = 2 < 128, n=2_000_000 > 1_000_000 → ortools
    assert select_solver(n=2_000_000, m=2_000_000, nnz=4_000_000) == 'ortools'


def test_boundary_n_just_below_lemon():
    # k=10 < 128, n=999_999 < 1_000_000 → lemon
    assert select_solver(n=999_999, m=10, nnz=999_999 * 10) == 'lemon'


def test_boundary_n_just_above_ortools():
    # k=10 < 128, n=1_000_001 > 1_000_000 → ortools
    assert select_solver(n=1_000_001, m=10, nnz=1_000_001 * 10) == 'ortools'


def test_override_bonneel_beats_lemon_routing():
    # k=10 would normally → lemon, but override wins
    assert select_solver(n=10, m=10, nnz=100, solver='bonneel') == 'bonneel'


def test_override_lemon_beats_bonneel_routing():
    # k=200 would normally → bonneel, but override wins
    assert select_solver(n=10, m=10, nnz=2000, solver='lemon') == 'lemon'


def test_override_ortools():
    assert select_solver(n=10, m=10, nnz=100, solver='ortools') == 'ortools'


def test_large_k_fully_dense_routes_to_bonneel():
    # n=200, k=200 > 128 → bonneel
    assert select_solver(n=200, m=200, nnz=200 * 200) == 'bonneel'


def test_small_n_fully_dense_routes_to_lemon():
    # n=10, k=10 < 128 → lemon
    assert select_solver(n=10, m=10, nnz=10 * 10) == 'lemon'
