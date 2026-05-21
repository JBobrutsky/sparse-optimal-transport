"""Warm-start refinement for sparse-OT (see docs/refinement.md)."""
from __future__ import annotations

import numpy as np
import scipy.sparse

import warnings

from sparse_ot.feasibility import check_feasibility
from sparse_ot.sparse_utils import to_csr

_MARGINAL_TOL = 1e-6


def _parse_warm_start(warm_start, n, m):
    """Normalize ``warm_start`` to ``(G_csr, u, v)``.

    Accepted forms:
      * ``(G, info)`` where ``info`` is a dict with keys ``u`` and ``v``.
      * ``(G, u, v)`` bare 3-tuple.

    ``G`` may be a ``scipy.sparse`` matrix (any format) or a 2-D ``ndarray``.
    Both are normalized to CSR.
    """
    if not isinstance(warm_start, tuple) or len(warm_start) not in (2, 3):
        raise TypeError(
            "warm_start must be a 2-tuple (G, info) or a 3-tuple (G, u, v); "
            f"got {type(warm_start).__name__} of length "
            f"{len(warm_start) if hasattr(warm_start, '__len__') else '?'}"
        )

    if len(warm_start) == 2:
        G, info = warm_start
        if not isinstance(info, dict):
            raise TypeError(
                "warm_start[1] must be the info dict from a prior "
                f"emd(..., log=True) call; got {type(info).__name__}"
            )
        if "u" not in info or "v" not in info:
            raise TypeError(
                "warm_start info dict must contain keys 'u' and 'v'; "
                f"got keys {sorted(info.keys())!r}"
            )
        u = info["u"]
        v = info["v"]
    else:
        G, u, v = warm_start

    if scipy.sparse.issparse(G):
        G_csr = G.tocsr().astype(np.float64)
    elif isinstance(G, np.ndarray) and G.ndim == 2:
        G_csr = scipy.sparse.csr_matrix(G.astype(np.float64, copy=False))
    else:
        raise TypeError(
            "warm_start G must be a CSR matrix or a 2-D ndarray; "
            f"got {type(G).__name__}"
        )

    if G_csr.shape != (n, m):
        raise ValueError(
            f"warm_start G has shape {G_csr.shape}; "
            f"expected ({n}, {m}) to match M_full"
        )

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 1:
        raise ValueError(
            f"warm_start u must be 1-D; got shape {u.shape}"
        )
    if v.ndim != 1:
        raise ValueError(
            f"warm_start v must be 1-D; got shape {v.shape}"
        )
    if len(u) != n:
        raise ValueError(
            f"warm_start len(u)={len(u)}; expected {n} to match M_full"
        )
    if len(v) != m:
        raise ValueError(
            f"warm_start len(v)={len(v)}; expected {m} to match M_full"
        )
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("warm_start u, v must be finite (no NaN or inf)")

    return G_csr, u, v


def _compute_reduced_costs(M_csr, u, v, tol=0.0):
    """Reduced cost ``M[i, j] - u[i] - v[j]`` over every nnz edge of M.

    Returns ``(rc, min_rc, n_violating)`` where ``n_violating`` counts edges
    with ``rc < -tol``.
    """
    indptr = M_csr.indptr
    indices = M_csr.indices
    data = M_csr.data
    n = M_csr.shape[0]
    row_idx = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
    rc = data - u[row_idx] - v[indices]
    if rc.size == 0:
        return rc, 0.0, 0
    min_rc = float(rc.min())
    n_viol = int(np.sum(rc < -tol))
    return rc, min_rc, n_viol


def _default_tol(M_csr):
    """Scale-relative tolerance for the dual-feasibility check."""
    if M_csr.nnz == 0:
        return 1e-9
    return 1e-9 * max(1.0, float(np.abs(M_csr.data).max()))


def _drop_explicit_zeros(G_warm_csr):
    """Return a copy of G_warm with explicit zero entries removed.

    Called on the already-optimal branch, where G_warm is already known to
    have its support contained in M_csr's (by ``_verify_support_subset``).
    The returned CSR keeps G_warm's index layout, not M_csr's, but the math
    that follows (cost via ``G.multiply(M_csr).sum()``, marginal sums) is
    layout-independent.
    """
    G = G_warm_csr.copy()
    G.eliminate_zeros()
    return G


def _check_marginals_csr(G, a, b):
    row_sum = np.asarray(G.sum(axis=1)).ravel()
    col_sum = np.asarray(G.sum(axis=0)).ravel()
    err_a = float(np.max(np.abs(row_sum - a)))
    err_b = float(np.max(np.abs(col_sum - b)))
    return err_a, err_b


def _verify_support_subset(G_warm_csr, M_csr):
    """Raise if G_warm has a nonzero outside M_csr's stored support.

    Vectorized: encodes (row, col) as int64 keys row*m + col, sorts M's
    keys once, then np.searchsorted to check each G nonzero. O(nnz log nnz)
    in pure NumPy.
    """
    G_coo = G_warm_csr.tocoo()
    nz = G_coo.data != 0.0
    if not np.any(nz):
        return
    n, m = M_csr.shape
    g_keys = (G_coo.row[nz].astype(np.int64) * m
              + G_coo.col[nz].astype(np.int64))

    M_coo = M_csr.tocoo()
    m_keys = (M_coo.row.astype(np.int64) * m
              + M_coo.col.astype(np.int64))
    m_keys.sort()

    idx = np.searchsorted(m_keys, g_keys)
    # idx == len means key > all; clip to safe index then check equality
    in_range = idx < m_keys.size
    found = np.zeros_like(g_keys, dtype=bool)
    found[in_range] = m_keys[idx[in_range]] == g_keys[in_range]

    if not np.all(found):
        bad = np.where(~found)[0][0]
        r = int(G_coo.row[nz][bad])
        c = int(G_coo.col[nz][bad])
        raise ValueError(
            f"warm_start G has a nonzero at ({r}, {c}) which is not in "
            f"M_full's support; warm_start is incompatible with M_full"
        )


def refine_from_warm_start(a, b, M_csr, warm_start, *,
                           numItermax, log, center_dual, reduced_cost_tol):
    n, m = M_csr.shape

    G_warm, u, v = _parse_warm_start(warm_start, n, m)
    _verify_support_subset(G_warm, M_csr)

    # Feasibility precondition on M_full (same as cold path).
    row_ptr, col_idx, _costs, _n, _m, _k = to_csr(M_csr, 0.0)
    check_feasibility(a, b, row_ptr, col_idx)

    tol = _default_tol(M_csr) if reduced_cost_tol is None else float(reduced_cost_tol)

    rc, min_rc, n_viol = _compute_reduced_costs(M_csr, u, v, tol=tol)

    if min_rc >= -tol:
        G = _drop_explicit_zeros(G_warm)
        refine_info = {
            "warm_start_optimal": True,
            "num_passes": 0,
            "initial_min_reduced_cost": min_rc,
            "edges_added": 0,
        }
    else:
        # v1 fallback: cold re-solve on M_full. The C++ pybind binding does
        # not yet accept (u0, v0) for true basis warm-start; see spec
        # "Pybind / C++ extension" and "Open questions". The verifier above
        # is still useful -- it confirms when no re-solve is needed at all.
        # The cold re-solve produces the optimum on (a, b, M_full).
        from sparse_ot._ext import _bonneel
        row_ptr_full, col_idx_full, costs_full, _, _, k_full = to_csr(M_csr, 0.0)
        if numItermax is None:
            num_iter = min(50_000_000, max(100_000, 100 * (n + m + k_full)))
        else:
            num_iter = int(numItermax)
        rows_out, cols_out, vals_out, u, v = _bonneel.solve_sparse(
            a, b, row_ptr_full, col_idx_full, costs_full, num_iter
        )
        G = scipy.sparse.csr_matrix(
            (vals_out, (rows_out, cols_out)), shape=(n, m)
        )
        edges_added = int(G.nnz - G_warm.nnz)
        refine_info = {
            "warm_start_optimal": False,
            "num_passes": 1,
            "initial_min_reduced_cost": min_rc,
            "edges_added": max(edges_added, 0),
        }

    if center_dual:
        shift = float(u.mean())
        u = u - shift
        v = v + shift

    err_a, err_b = _check_marginals_csr(G, a, b)
    converged = max(err_a, err_b) <= _MARGINAL_TOL
    warn_msg = None
    if not converged:
        warn_msg = (
            f"marginals not satisfied after warm-start refinement: "
            f"|G.sum(1)-a|={err_a:.2e}, |G.sum(0)-b|={err_b:.2e} "
            f"(tol={_MARGINAL_TOL:.0e})."
        )
        warnings.warn(warn_msg, RuntimeWarning, stacklevel=3)

    if log:
        cost = float(G.multiply(M_csr).sum())
        return G, {
            "cost": cost,
            "u": u,
            "v": v,
            "warning": warn_msg,
            "result_code": 1 if converged else 0,
            "refine": refine_info,
        }
    return G
