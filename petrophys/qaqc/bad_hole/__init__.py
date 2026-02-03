"""
Bad Hole Detection Module.

This module provides tools for detecting bad hole conditions
including washouts, caving, and other borehole irregularities.
"""

from petrophys.qaqc.bad_hole.washout_detection import (
    WashoutDetection,
    detect_washout,
)
from petrophys.qaqc.bad_hole.flag_generator import (
    generate_bad_hole_flags,
    generate_washout_flag,
    generate_large_hole_flag,
    generate_low_density_flag,
    generate_pef_exclusion_flag,
)
from petrophys.qaqc.bad_hole.curve_masking import (
    mask_bad_hole_curves,
    apply_quality_flags,
)

__all__ = [
    "WashoutDetection",
    "detect_washout",
    "generate_bad_hole_flags",
    "generate_washout_flag",
    "generate_large_hole_flag",
    "generate_low_density_flag",
    "generate_pef_exclusion_flag",
    "mask_bad_hole_curves",
    "apply_quality_flags",
]
