import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.feasibility import check_feasibility


def emd(a, b, M, numItermax=100000, log=False, center_dual=True):
    """Transport plan between distributions a and b with cost matrix M.

    POT-compatible: drop-in for ``ot.emd``. Dense numpy ``M`` returns a dense
    ndarray; ``scipy.sparse`` ``M`` returns a CSR.

    Parameters
    ----------
    a, b : array-like
    M    : ndarray (n, m) or scipy.sparse (n, m)
    numItermax  : int
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
        row_ptr, col_idx, costs, n, m, _ = to_csr(M, 0.0)
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        check_feasibility(a, b, row_ptr, col_idx)
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
        G, u, v = _bonneel.solve_dense(a, b, M_dense, numItermax)
        M_for_cost = M_dense

    if center_dual:
        shift = float(u.mean())
        u = u - shift
        v = v + shift

    if log:
        if scipy.sparse.issparse(G):
            cost = float(G.multiply(M_for_cost).sum())
        else:
            cost = float(np.sum(G * M_for_cost))
        return G, {"cost": cost, "u": u, "v": v,
                   "warning": None, "result_code": 1}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False):
    """POT-compatible ``ot.emd2``."""
    G, info = emd(a, b, M, numItermax=numItermax, log=True)
    cost = info["cost"]
    if return_matrix:
        info = {**info, "G": G}
        return (cost, info) if log else (cost, G)
    return (cost, info) if log else cost
