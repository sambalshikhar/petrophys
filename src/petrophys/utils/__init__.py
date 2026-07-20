"""Compatibility namespace for internal numerical helpers."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys._internal",
    {
        "analysis_helpers": "petrophys._internal.analysis_helpers",
        "interpolation": "petrophys._internal.interpolation",
        "statistics": "petrophys._internal.statistics",
        "validators": "petrophys._internal.validators",
    },
)

from petrophys._internal.analysis_helpers import (
    resolve_well_depth_range,
    well_depth_array,
)
from petrophys._internal.interpolation import (
    fill_gaps,
    linear_interpolate,
    log_interpolate,
    resample_curve,
)
from petrophys._internal.statistics import (
    finite_min_max,
    geometric_mean,
    harmonic_mean,
    percentile_stats,
    trim_mean,
    weighted_average,
)
from petrophys._internal.validators import (
    validate_array,
    validate_curve,
    validate_range,
    validate_well,
)

__all__ += [
    "fill_gaps",
    "finite_min_max",
    "geometric_mean",
    "harmonic_mean",
    "linear_interpolate",
    "log_interpolate",
    "percentile_stats",
    "resample_curve",
    "resolve_well_depth_range",
    "trim_mean",
    "validate_array",
    "validate_curve",
    "validate_range",
    "validate_well",
    "weighted_average",
    "well_depth_array",
]
