"""The mineral solvers' non-negative least squares reaches the optimum on every
scipy version (some scipy releases stop short of it on these small systems)."""

import itertools
import unittest

import _support  # noqa: F401  (import path)
import numpy as np

from petrophys._internal.nnls import nnls

# Unity, density, Pe x density (U) and sonic rows for quartz, calcite,
# dolomite and pore fluid: the system the mineral solver builds per sample.
MINERAL_SYSTEM = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [2.65, 2.71, 2.87, 1.0],
    [4.77, 13.821, 8.897, 0.36],
    [55.5, 47.6, 43.5, 189.0],
])


def brute_force(A, b):
    """Smallest residual over every set of non-negative active variables."""
    best = np.linalg.norm(b)
    for size in range(1, A.shape[1] + 1):
        for support in itertools.combinations(range(A.shape[1]), size):
            solution = np.linalg.lstsq(A[:, support], b, rcond=None)[0]
            if np.all(solution >= 0):
                x = np.zeros(A.shape[1])
                x[list(support)] = solution
                best = min(best, np.linalg.norm(A @ x - b))
    return best


class NnlsTests(unittest.TestCase):
    def assert_optimal(self, A, b):
        x, residual = nnls(A, b)
        self.assertTrue(np.all(x >= 0))
        self.assertAlmostEqual(residual, float(np.linalg.norm(A @ x - b)), places=9)
        self.assertLessEqual(residual, brute_force(A, b) + 1e-9)

    def test_mineral_systems_from_log_responses(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            fractions = rng.dirichlet(np.ones(4)) * rng.uniform(0.6, 1.0)
            noise = rng.normal(0, [0.05, 0.08, 1.5, 15.0])
            b = MINERAL_SYSTEM @ fractions + noise
            self.assert_optimal(MINERAL_SYSTEM, b)

    def test_random_small_systems(self):
        rng = np.random.default_rng(11)
        for _ in range(300):
            rows, columns = rng.integers(3, 6), rng.integers(2, 5)
            A = rng.normal(size=(rows, columns)) * rng.uniform(0.5, 50)
            b = rng.normal(size=rows) * 10
            self.assert_optimal(A, b)


if __name__ == "__main__":
    unittest.main()
