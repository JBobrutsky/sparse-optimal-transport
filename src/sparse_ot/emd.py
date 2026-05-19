import warnings

import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.feasibility import check_feasibility

# Convergence tolerance for the post-solve marginal check. Bonneel's network
# simplex terminates at numItermax without raising; if it stops early the
# returned flows can violate row/col marginals by orders of magnitude more
# than machine epsilon. Anything above this is treated as non-convergence.
_MARGINAL_TOL = 1e-6


def _default_num_iter(n, m, k):
    # Network simplex empirically converges in O((n+m) * sqrt(k)) pivots on
    # well-behaved OT problems. Pick a generous linear multiple of the problem
    # size so neither small nor large instances truncate. Capped to keep
    # pathological inputs from running unboundedly.
    return min(50_000_000, max(100_000, 100 * (n + m + k)))


def _check_marginals(G, a, b):
    if scipy.sparse.issparse(G):
        row_sum = np.asarray(G.sum(axis=1)).ravel()
        col_sum = np.asarray(G.sum(axis=0)).ravel()
    else:
        row_sum = G.sum(axis=1)
        col_sum = G.sum(axis=0)
    err_a = float(np.max(np.abs(row_sum - a)))
    err_b = float(np.max(np.abs(col_sum - b)))
    return err_a, err_b


def emd(a, b, M, numItermax=None, log=False, center_dual=True):
    """Transport plan between distributions a and b with cost matrix M.

    POT-compatible: drop-in for ``ot.emd``. Dense numpy ``M`` returns a dense
    ndarray; ``scipy.sparse`` ``M`` returns a CSR.

    Parameters
    ----------
    a, b : array-like
    M    : ndarray (n, m) or scipy.sparse (n, m)
    numItermax  : int or None — pivot-iteration cap for the network simplex.
                  ``None`` picks a problem-size-aware default.
    log         : bool — if True, return ``(G, info)`` with keys
                  ``cost, u, v, warning, result_code``.
    center_dual : bool — if True, shift u/v so u has zero mean while
                  preserving u[i] + v[j].
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / a.sum()
    b = b / b.sum()

    if scipy.sparse.issparse(M):
        row_ptr, col_idx, costs, n, m, k = to_csr(M, 0.0)
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        check_feasibility(a, b, row_ptr, col_idx)
        if numItermax is None:
            numItermax = _default_num_iter(n, m, k)
        rows, cols, vals, u, v = _bonneel.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
        G = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
        M_for_cost = M
    else:
        M_dense = np.ascontiguousarray(M, dtype=np.float64)
        n, m = M_dense.shape
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        if numItermax is None:
            numItermax = _default_num_iter(n, m, n * m)
        G, u, v = _bonneel.solve_dense(a, b, M_dense, numItermax)
        M_for_cost = M_dense

    if center_dual:
        shift = float(u.mean())
        u = u - shift
        v = v + shift

    err_a, err_b = _check_marginals(G, a, b)
    converged = max(err_a, err_b) <= _MARGINAL_TOL
    if not converged:
        msg = (
            f"network simplex did not converge: |G.sum(0)-b|={err_b:.2e}, "
            f"|G.sum(1)-a|={err_a:.2e} (tol={_MARGINAL_TOL:.0e}). "
            f"Try a larger numItermax (current={numItermax})."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if log:
        if scipy.sparse.issparse(G):
            cost = float(G.multiply(M_for_cost).sum())
        else:
            cost = float(np.sum(G * M_for_cost))
        return G, {
            "cost": cost,
            "u": u,
            "v": v,
            "warning": None if converged else msg,
            "result_code": 1 if converged else 0,
        }
    return G


def emd2(a, b, M, numItermax=None, log=False, return_matrix=False):
    """POT-compatible ``ot.emd2``."""
    G, info = emd(a, b, M, numItermax=numItermax, log=True)
    cost = info["cost"]
    if return_matrix:
        info = {**info, "G": G}
        return (cost, info) if log else (cost, G)
    return (cost, info) if log else cost
