import numpy as np

from sparse_ot._ext import _bonneel


def emd(a, b, M, numItermax=100000, log=False, center_dual=True,
        cost_sparsity_threshold=0.0, solver=None):
    """Transport plan between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd(). Returns a dense numpy array.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like, shape (n, m)
    numItermax : int
        Max iterations for the solver.
    log : bool
        If True, return (G, log_dict).
    center_dual : bool
        Accepted for POT compatibility; has no effect.
    cost_sparsity_threshold : float
        Values with |M[i,j]| <= threshold are treated as absent edges. Not yet used in this version.
    solver : str or None
        'bonneel' or None (auto).

    Returns
    -------
    G : ndarray, shape (n, m)
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64, order="C")

    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    if M.shape != (len(a), len(b)):
        raise ValueError(
            f"M must have shape ({len(a)}, {len(b)}), got {M.shape}"
        )
    if solver not in (None, "bonneel"):
        raise ValueError(
            f"solver={solver!r} not available in Plan 1. "
            "Use None or 'bonneel'."
        )

    # Bonneel requires exactly balanced supply/demand.
    a = a / a.sum()
    b = b / b.sum()

    G = _bonneel.solve_dense(a, b, M, numItermax)

    if log:
        return G, {}
    return G


def emd2(a, b, M, numItermax=100000, log=False, return_matrix=False,
         cost_sparsity_threshold=0.0, solver=None):
    """OT cost between distributions a and b with cost matrix M.

    Drop-in replacement for ot.emd2(). Returns a float scalar.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)
    M : array-like, shape (n, m)
    numItermax : int
    log : bool
        If True, return (cost, log_dict) or (cost, G, log_dict) with return_matrix.
    return_matrix : bool
        If True, also return the transport plan G.
    cost_sparsity_threshold : float
    solver : str or None

    Returns
    -------
    cost : float
    G : ndarray, shape (n, m)  — only if return_matrix=True
    """
    G = emd(a, b, M, numItermax=numItermax, log=False,
            cost_sparsity_threshold=cost_sparsity_threshold, solver=solver)

    M_n = np.asarray(M, dtype=np.float64)
    cost = float(np.sum(G * M_n))

    if return_matrix:
        if log:
            return cost, G, {}
        return cost, G
    if log:
        return cost, {}
    return cost
