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
    finite_min_max,
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
from petrophys.utils.analysis_helpers import (
    resolve_well_depth_range,
    well_depth_array,
)

__all__ = [
    # Interpolation
    "linear_interpolate",
    "log_interpolate",
    "resample_curve",
    "fill_gaps",
    # Statistics
    "finite_min_max",
    "weighted_average",
    "harmonic_mean",
    "geometric_mean",
    "trim_mean",
    "percentile_stats",
    "resolve_well_depth_range",
    "well_depth_array",
    # Validators
    "validate_array",
    "validate_range",
    "validate_curve",
    "validate_well",
]
