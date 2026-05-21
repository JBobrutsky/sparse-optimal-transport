"""Warm-start refinement for sparse-OT (see docs/refinement.md)."""
from __future__ import annotations

import numpy as np
import scipy.sparse


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

    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
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


def refine_from_warm_start(a, b, M_csr, warm_start, *,
                           numItermax, log, center_dual, reduced_cost_tol):
    raise NotImplementedError("filled in by later tasks")
