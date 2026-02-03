"""
Utility functions for petrophysics calculations.
"""

from petrophys.utils.interpolation import (
    linear_interpolate,
    log_interpolate,
    resample_curve,
    fill_gaps,
)
from petrophys.utils.statistics import (
    weighted_average,
    harmonic_mean,
    geometric_mean,
    trim_mean,
    percentile_stats,
)
from petrophys.utils.validators import (
    validate_array,
    validate_range,
    validate_curve,
    validate_well,
)

__all__ = [
    # Interpolation
    "linear_interpolate",
    "log_interpolate",
    "resample_curve",
    "fill_gaps",
    # Statistics
    "weighted_average",
    "harmonic_mean",
    "geometric_mean",
    "trim_mean",
    "percentile_stats",
    # Validators
    "validate_array",
    "validate_range",
    "validate_curve",
    "validate_well",
]
