import warnings

import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr, _default_num_iter, bonneel_sparse_solve, _MARGINAL_TOL
from sparse_ot.feasibility import check_feasibility



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


def emd(a, b, M, numItermax=None, log=False, center_dual=True,
        warm_start=None, reduced_cost_tol=None):
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
    warm_start : tuple or None — if provided, refine to optimum on M from the
                  given prior solve. Form: ``(G_warm, info)`` where info is the
                  log dict from a previous ``emd(..., log=True)`` call, or the
                  bare 3-tuple ``(G_warm, u, v)``. ``G_warm`` may be CSR or a
                  2-D ndarray; both forms accepted. Only supported when M is
                  CSR. See ``docs/refinement.md``.
    reduced_cost_tol : float or None — tolerance for the dual-feasibility check.
                  None picks ``1e-9 * max(1, |M|_inf)``.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / a.sum()
    b = b / b.sum()

    if warm_start is not None:
        if not scipy.sparse.issparse(M):
            raise NotImplementedError(
                "warm_start is only supported for sparse (CSR) M in v1; "
                "for dense M_full the cold path is already optimal."
            )
        # Lazy import to avoid a refine.py <-> emd.py cycle.
        from sparse_ot.refine import refine_from_warm_start
        return refine_from_warm_start(
            a, b, M, warm_start,
            numItermax=numItermax, log=log,
            center_dual=center_dual,
            reduced_cost_tol=reduced_cost_tol,
        )

    if scipy.sparse.issparse(M):
        row_ptr, col_idx, costs, n, m, k = to_csr(M, 0.0)
        if (len(a), len(b)) != (n, m):
            raise ValueError(
                f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
            )
        density = k / (n * m) if n * m > 0 else 0.0
        if density > 0.5:
            warnings.warn(
                f"M is CSR but {density:.1%} dense ({k} nnz of {n*m} entries); "
                f"the sparse path pays CSR indirection overhead with no "
                f"sparsity benefit. Pass M.toarray() to use the dense "
                f"Bonneel path for faster solves.",
                RuntimeWarning,
                stacklevel=2,
            )
        check_feasibility(a, b, row_ptr, col_idx)
        G, u, v = bonneel_sparse_solve(a, b, row_ptr, col_idx, costs, n, m, numItermax)
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
            f"marginals not satisfied: |G.sum(1)-a|={err_a:.2e}, "
            f"|G.sum(0)-b|={err_b:.2e} (tol={_MARGINAL_TOL:.0e}). "
            f"This usually means the sparse support cannot accommodate the "
            f"given marginals (no feasible plan exists on those edges). "
            f"Provide a denser cost matrix or accept the approximate flow. "
            f"Less commonly, this can also mean numItermax was hit too early "
            f"(current={numItermax}); rerun with a larger value to rule that out."
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
