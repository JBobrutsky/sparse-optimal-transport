# src/sparse_ot/emd.py
import numpy as np
import scipy.sparse

from sparse_ot._ext import _bonneel
from sparse_ot.sparse_utils import to_csr
from sparse_ot.routing import select_solver


def emd(a, b, M, numItermax=100000, log=False, center_dual=True,
        cost_sparsity_threshold=0.0, solver=None,
        ortools_cost_scale=1e6):
    """Transport plan between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd(). Dense numpy input returns a dense numpy
    array; scipy sparse input returns scipy CSR.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like (n, m) or scipy sparse (n, m)
    numItermax : int
    log : bool — if True, return (G, log_dict)
    center_dual : bool — accepted for POT compatibility; not used
    cost_sparsity_threshold : float — dense M only: drop |M[i,j]| <= threshold
    solver : str or None — 'bonneel', 'lemon', 'ortools', or None (auto)
    ortools_cost_scale : float — int64 scale applied to float costs by OR-Tools

    Returns
    -------
    G : ndarray (n, m) or scipy CSR (n, m)
    (G, {}) if log=True
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if solver not in (None, 'bonneel', 'lemon', 'ortools'):
        raise ValueError(
            f"solver={solver!r} must be None, 'bonneel', 'lemon', or 'ortools'"
        )

    dense_input = not scipy.sparse.issparse(M)
    row_ptr, col_idx, costs, n, m, nnz = to_csr(M, cost_sparsity_threshold)

    if (len(a), len(b)) != (n, m):
        raise ValueError(
            f"M must have shape ({len(a)}, {len(b)}), got ({n}, {m})"
        )

    a = a / a.sum()
    b = b / b.sum()

    selected = select_solver(n, m, nnz, solver)

    if selected == 'bonneel' and not dense_input:
        # M.toarray() fills absent cells with 0, which Bonneel cannot
        # distinguish from real zero-cost edges — it routes all mass through
        # the "free" absent cells and returns a degenerate zero-cost plan.
        raise ValueError(
            "Bonneel does not support sparse cost matrices: M.toarray() would "
            "fill absent edges with 0, which Bonneel treats as free edges and "
            "routes through silently. Use solver='lemon' or solver='ortools' "
            "for sparse M, or pass a dense numpy array with +inf in absent cells."
        )

    # Feasibility check (spec §2/§3): sparse-input paths only. By this point
    # Bonneel + sparse has already been refused above.
    if not dense_input:
        from sparse_ot.feasibility import check_feasibility
        check_feasibility(a, b, row_ptr, col_idx)

    if selected == 'bonneel':
        M_dense = np.asarray(M, dtype=np.float64, order='C')
        G = _bonneel.solve_dense(a, b, M_dense, numItermax)
        if log:
            return G, {}
        return G

    if selected == 'lemon':
        from sparse_ot._ext import _lemon
        rows, cols, vals = _lemon.solve_sparse(
            a, b, row_ptr, col_idx, costs, numItermax
        )
    else:
        # selected == 'ortools'
        from sparse_ot.ortools_solver import solve_ortools
        rows, cols, vals = solve_ortools(
            a, b, row_ptr, col_idx, costs,
            ortools_cost_scale=ortools_cost_scale,
        )

    G_sp = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    G = G_sp.toarray() if dense_input else G_sp

    if log:
        return G, {}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
         cost_sparsity_threshold=0.0, solver=None,
         ortools_cost_scale=1e6):
    """OT cost between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd2(). Returns a float scalar.
    """
    G = emd(a, b, M, numItermax=numItermax, log=False,
            cost_sparsity_threshold=cost_sparsity_threshold,
            solver=solver, ortools_cost_scale=ortools_cost_scale)

    if scipy.sparse.issparse(M):
        M_arr = np.asarray(M.toarray(), dtype=np.float64)
    else:
        M_arr = np.asarray(M, dtype=np.float64)

    if scipy.sparse.issparse(G):
        cost = float(G.multiply(M_arr).sum())
    else:
        cost = float(np.sum(G * M_arr))

    if return_matrix:
        if log:
            return cost, G, {}
        return cost, G
    if log:
        return cost, {}
    return cost
