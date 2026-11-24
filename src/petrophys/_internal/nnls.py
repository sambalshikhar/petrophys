"""Non-negative least squares that returns the optimum on every scipy version.

``scipy.optimize.nnls`` in some releases (1.15.3 among those tested) stops
short of the optimum on small, well-conditioned systems such as the per-sample
mineral systems, and reports a residual other than that of the solution it
returns, and older releases (1.13 among those tested) raise RuntimeError when
they reach their iteration limit. Its answer is kept when it satisfies the
optimality (KKT) conditions; otherwise the system, a handful of unknowns, is
solved exactly by trying every set of active variables. The residual returned is always that of ``x``.
"""

import math
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls as _scipy_nnls


def _scipy_solution(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """scipy's ``(x, residual)``; NaN when it stops at its iteration limit."""
    try:
        x, reported = _scipy_nnls(A, b)
    except RuntimeError:
        return np.full(A.shape[1], np.nan), math.nan
    return x, float(reported)


def _is_optimal(A: np.ndarray, b: np.ndarray, x: np.ndarray, a_max: Optional[float] = None) -> bool:
    """KKT conditions of min ||Ax - b|| subject to x >= 0, to rounding.

    ``a_max``, the largest ``|A|``, may be given when many systems share ``A``.
    """
    # Plain Python on these few numbers: abs, max and comparisons are exact,
    # and far cheaper than numpy calls on arrays this small.
    xs = x.tolist()
    if not all(math.isfinite(v) and v >= 0 for v in xs):
        return False
    gradient = (A.T @ (A @ x - b)).tolist()
    if a_max is None:
        a_max = float(np.abs(A).max()) if A.size else 0.0
    # Rounding of A.T (A x - b) is of order eps * |A| (|A| |x| + |b|); allow
    # ten thousand times that, far below what a suboptimal step leaves.
    scale = 1e-12 * max(1.0, a_max * (a_max * (max(abs(v) for v in xs) + 1.0) + max(abs(v) for v in b.tolist())))
    return all(abs(g) <= scale if v > 0 else g >= -scale for g, v in zip(gradient, xs))


def _exact_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimum by enumerating active sets; for systems of a few unknowns."""
    n = A.shape[1]
    best_x, best_norm = np.zeros(n), float(np.linalg.norm(b))
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            columns = list(support)
            solution = np.linalg.lstsq(A[:, columns], b, rcond=None)[0]
            if np.any(solution < 0):
                continue
            x = np.zeros(n)
            x[columns] = solution
            norm = float(np.linalg.norm(A @ x - b))
            if norm < best_norm:
                best_x, best_norm = x, norm
    return best_x


def nnls_many(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``x`` of :func:`nnls` for each row of ``B`` against one float ``A``.

    The optimality check of :func:`_is_optimal` runs on all the solutions at
    once; a system that fails it is solved exactly.
    """
    X = np.array([_scipy_solution(A, b)[0] for b in B]).reshape(len(B), A.shape[1])
    if not len(B):
        return X
    a_max = float(np.abs(A).max()) if A.size else 0.0
    gradient = (X @ A.T - B) @ A
    scale = 1e-12 * np.maximum(1.0, a_max * (a_max * (np.abs(X).max(axis=1) + 1.0) + np.abs(B).max(axis=1)))
    free = X > 0
    within = np.where(free, np.abs(gradient) <= scale[:, None], gradient >= -scale[:, None])
    optimal = np.isfinite(X).all(axis=1) & (X >= 0).all(axis=1) & within.all(axis=1)
    for j in np.flatnonzero(~optimal):
        X[j] = _exact_nnls(A, B[j])
    return X


def nnls(A, b) -> Tuple[np.ndarray, float]:
    """``(x, ||Ax - b||)`` with ``x >= 0`` minimising the residual."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x, reported = _scipy_solution(A, b)
    if not _is_optimal(A, b, x):
        x = _exact_nnls(A, b)
    residual = float(np.linalg.norm(A @ x - b))
    # Keep scipy's figure when it is that of its solution (bit-identical results).
    if abs(float(reported) - residual) <= 1e-12 * (1.0 + residual):
        residual = float(reported)
    return x, residual
