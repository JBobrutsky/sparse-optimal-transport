# benchmarks/problems.py
"""Reproducible k-NN grid OT problems.

Source and target nodes are placed on a regular 1D grid at positions 0..n-1.
Each source i connects to up to `k` target nodes nearest to i (clipped at
grid boundaries) with cost (i - j) ** 2. Distributions a and b are
Dirichlet(1, ..., 1) draws (full-support, reproducible per `seed`).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse


def generate_knn_grid_problem(
    n: int, k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, scipy.sparse.csr_matrix]:
    """Return (a, b, M) where M is an (n, n) scipy CSR of squared-distance costs."""
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))

    half = k // 2
    rows, cols, data = [], [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        for j in range(lo, hi):
            rows.append(i)
            cols.append(j)
            data.append(float((i - j) ** 2))

    M = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32),
          np.asarray(cols, dtype=np.int32))),
        shape=(n, n),
    )
    return a, b, M
