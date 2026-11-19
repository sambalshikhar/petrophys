"""Small, shared checks for numerical API inputs and diagnostic outputs."""

import warnings

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.exceptions import DataQualityWarning


def numeric_array(value, name):
    """Read numeric input, naming the argument if conversion fails."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values; got nonnumeric input") from exc


def warn_invalid(mask, name, expectation, action):
    """Warn once for the selected samples; ordinary missing values are caller-defined."""
    if reproducing_benchmark():
        return
    bad = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(bad))
    if not count:
        return
    positions = np.flatnonzero(bad)[:5].tolist()
    warnings.warn(
        f"{name}: {count} of {bad.size} value(s) {expectation} "
        f"(zero-based flat indices {positions}). {action}",
        DataQualityWarning,
        stacklevel=3,
    )


def warn_range(values, name, lower=None, upper=None, *, open_lower=False, action="Values are used unchanged; review the result."):
    """Warn on finite out-of-range values and infinity, but not missing NaN."""
    if reproducing_benchmark():
        return
    array = numeric_array(values, name)
    bad = np.isinf(array)
    if lower is not None:
        bad = bad | (array <= lower if open_lower else array < lower)
    if upper is not None:
        bad = bad | (array > upper)
    bounds = "must be finite"
    if lower is not None:
        bounds += f" and {'greater than' if open_lower else 'at least'} {lower:g}"
    if upper is not None:
        bounds += f" and at most {upper:g}"
    warn_invalid(bad, name, bounds, action)


def broadcast_inputs(**inputs):
    """Convert numerical inputs and reject incompatible sample shapes clearly."""
    arrays = [numeric_array(value, name) for name, value in inputs.items()]
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in zip(inputs, arrays))
        raise ValueError(f"Inputs must have compatible sample shapes; got {shapes}") from exc
