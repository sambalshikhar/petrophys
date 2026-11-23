"""Non-negative least squares that returns the optimum on every scipy version.

``scipy.optimize.nnls`` in some releases (1.15.3 among those tested) stops
short of the optimum on small, well-conditioned systems such as the per-sample
mineral systems, and reports a residual other than that of the solution it
returns. Its answer is kept when it satisfies the optimality (KKT) conditions;
otherwise the system, a handful of unknowns, is solved exactly by trying every
set of active variables. The residual returned is always that of ``x``.
"""

from itertools import combinations
from typing import Tuple

import numpy as np
from scipy.optimize import nnls as _scipy_nnls


def _is_optimal(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
    """KKT conditions of min ||Ax - b|| subject to x >= 0, to rounding."""
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        return False
    gradient = A.T @ (A @ x - b)
    a_max = float(np.abs(A).max()) if A.size else 0.0
    # Rounding of A.T (A x - b) is of order eps * |A| (|A| |x| + |b|); allow
    # ten thousand times that, far below what a suboptimal step leaves.
    scale = 1e-12 * max(1.0, a_max * (a_max * (float(np.abs(x).max()) + 1.0) + float(np.abs(b).max())))
    free = x > 0
    return bool(np.all(np.abs(gradient[free]) <= scale) and np.all(gradient[~free] >= -scale))


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


def nnls(A, b) -> Tuple[np.ndarray, float]:
    """``(x, ||Ax - b||)`` with ``x >= 0`` minimising the residual."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x, reported = _scipy_nnls(A, b)
    if not _is_optimal(A, b, x):
        x = _exact_nnls(A, b)
    residual = float(np.linalg.norm(A @ x - b))
    # Keep scipy's figure when it is that of its solution (bit-identical results).
    if abs(float(reported) - residual) <= 1e-12 * (1.0 + residual):
        residual = float(reported)
    return x, residual
