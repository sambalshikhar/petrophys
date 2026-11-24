"""
Quality checks and borehole-condition diagnostics.

This umbrella module contains tools for:
- Bad hole detection (washout, flags)
- Plausibility-range checks
- Held (stuck or padded) tool readings
- Curves off depth against each other
"""
from petrophys.preprocessing.quality.bad_hole import (
    WashoutDetection,
    apply_quality_flags,
    detect_washout,
    generate_bad_hole_flags,
    generate_large_hole_flag,
    generate_low_density_flag,
    generate_pef_exclusion_flag,
    generate_washout_flag,
    mask_bad_hole_curves,
)
from petrophys.preprocessing.quality.depth_match import DepthMatch, depth_match
from petrophys.preprocessing.quality.depth_alignment import (
    LogDepthShift,
    find_log_depth_shift,
    shift_curves,
)
from petrophys.preprocessing.quality.held_readings import (
    HeldReadings,
    find_held_readings,
    held_reading_mask,
)
from petrophys.preprocessing.quality.value_ranges import (
    VALUE_RANGE_RULES,
    VALUE_RANGE_QC_PARAMETER,
    apply_value_range_qc,
    clean_curve_values,
    value_range_qc_note,
)
from petrophys.preprocessing.quality.wellbore import WellboreQC

__all__ = [
    # Bad Hole
    "WashoutDetection",
    "detect_washout",
    "generate_bad_hole_flags",
    "generate_washout_flag",
    "generate_large_hole_flag",
    "generate_low_density_flag",
    "generate_pef_exclusion_flag",
    "mask_bad_hole_curves",
    "apply_quality_flags",
    # Depth alignment
    "DepthMatch",
    "depth_match",
    "LogDepthShift",
    "find_log_depth_shift",
    "shift_curves",
    # Held readings
    "HeldReadings",
    "find_held_readings",
    "held_reading_mask",
    # Value Range QC
    "VALUE_RANGE_RULES",
    "VALUE_RANGE_QC_PARAMETER",
    "apply_value_range_qc",
    "clean_curve_values",
    "value_range_qc_note",
    "WellboreQC",
]
