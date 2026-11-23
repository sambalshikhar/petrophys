"""Interpretations of bundled wells reproduce their stored reference numbers.

A failure lists every quantity that moved. When the change is intended, run
``python tests/make_reference.py <scenario>`` and commit the updated JSON.
"""

import json
import math
import unittest

from reference_cases import SCENARIOS, reference_path, run

# Relative tolerance: absorbs floating-point differences between numpy and scipy
# versions and platforms, not changes of method or setting.
RTOL = 1e-6
ATOL = 1e-9


def differences(expected, actual, path=""):
    """Paths at which two JSON trees differ beyond the tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                found.append(f"{path}/{key}: missing")
            elif key not in expected:
                found.append(f"{path}/{key}: new")
            else:
                found += differences(expected[key], actual[key], f"{path}/{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} items -> {len(actual)}"]
        found = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            found += differences(left, right, f"{path}[{index}]")
        return found
    numbers = (int, float)
    if (isinstance(expected, numbers) and isinstance(actual, numbers)
            and not isinstance(expected, bool) and not isinstance(actual, bool)):
        if math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
            return []
        return [f"{path}: {expected!r} -> {actual!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} -> {actual!r}"]


class ReferenceValueTests(unittest.TestCase):
    def test_scenarios_reproduce_their_reference(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                path = reference_path(name)
                if not path.exists():
                    self.fail(f"No reference for {name}; run python tests/make_reference.py {name}")
                expected = json.loads(path.read_text(encoding="utf-8"))
                found = differences(expected, run(name))
                if found:
                    shown = "\n  ".join(found[:40])
                    more = f"\n  ... and {len(found) - 40} more" if len(found) > 40 else ""
                    self.fail(f"{name}: {len(found)} values moved\n  {shown}{more}\n"
                              f"If intended: python tests/make_reference.py {name}")


if __name__ == "__main__":
    unittest.main()
