"""Checks of the resistivity inputs shared by the water saturation equations.

A resistivity at or below zero has no physical meaning: a scalar parameter
such as Rw, Rmf or the clay resistivity raises ``ValueError``, and samples
with such a value in an array input (Rt, an Rw curve) give NaN saturation.
Inside the benchmark reproduction
(:func:`~petrophys._internal.benchmark.reproducing_benchmark`) the
equations keep the benchmark packages' calculation, which floored such
values at 1e-10 ohm.m.
"""

from typing import Any

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.validation import warn_range


def check_resistivity_parameters(**parameters: Any) -> None:
    """Raise ``ValueError`` for a scalar resistivity parameter at or below zero.

    Array parameters and None are left to :func:`nonpositive_resistivity`.
    """
    if reproducing_benchmark():
        return
    for name, value in parameters.items():
        if value is None or np.ndim(value) != 0:
            continue
        value = float(np.asarray(value, dtype=float))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive resistivity (ohm.m); got {value:g}")


def check_saturation_fractions(**fractions: Any) -> None:
    """Warn before fraction inputs are floored or clipped by a saturation model."""
    for name, value in fractions.items():
        warn_range(value, f"Saturation {name} (fraction)", 0, 1,
                   open_lower=name in {"phi", "phi_t"},
                   action="The model's existing floors and clipping are applied; review the input and result.")


def nonpositive_resistivity(*values: Any) -> np.ndarray:
    """True where any of ``values`` (scalars or arrays, broadcast) is at or below zero.

    Always False inside the benchmark reproduction.
    """
    mask = np.zeros(np.broadcast(*[np.asarray(value, dtype=float) for value in values]).shape, dtype=bool)
    if reproducing_benchmark():
        return mask
    for value in values:
        warn_range(value, "Saturation resistivity (ohm.m)", 0, open_lower=True,
                   action="Nonpositive samples give NaN saturation; review nonfinite resistivity measurements.")
        with np.errstate(invalid="ignore"):
            mask = mask | (np.asarray(value, dtype=float) <= 0.0)
    return mask
