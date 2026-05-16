# src/sparse_ot/routing.py
import json
from pathlib import Path

_THRESHOLDS_PATH = (
    Path(__file__).parent.parent.parent / "benchmarks" / "results" / "routing_thresholds.json"
)
_DEFAULTS = {"bonneel_lemon": 128, "lemon_ortools": 1_000_000}


def _load_thresholds():
    try:
        return json.loads(_THRESHOLDS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _DEFAULTS


def select_solver(n: int, m: int, nnz: int, solver: str | None = None) -> str:
    """Select the best solver for an OT problem.

    Parameters
    ----------
    n, m    : int   Source and target sizes.
    nnz     : int   Number of edges in the cost graph.
    solver  : str or None   Override ('bonneel', 'lemon', 'ortools') or None for auto.

    Returns
    -------
    'bonneel', 'lemon', or 'ortools'
    """
    if solver is not None:
        return solver
    thresholds = _load_thresholds()
    k = nnz / n if n > 0 else 0.0
    if k > thresholds["bonneel_lemon"]:
        return "bonneel"
    if n > thresholds["lemon_ortools"]:
        return "ortools"
    return "lemon"
