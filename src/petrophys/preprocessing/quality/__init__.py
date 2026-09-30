"""
Quality checks and borehole-condition diagnostics.

This umbrella module contains tools for:
- Bad hole detection (washout, flags)
- Plausibility-range checks
- Held (stuck or padded) tool readings
"""
from petrophys.preprocessing.quality.bad_hole import (
    WashoutDetection,
    detect_washout,
    generate_bad_hole_flags,
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
