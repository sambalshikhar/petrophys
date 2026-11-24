"""
Bad Hole Detection Module.

This module provides tools for detecting bad hole conditions
including washouts, caving, and other borehole irregularities.
"""

from petrophys.preprocessing.quality.bad_hole.washout_detection import (
    WashoutDetection,
    detect_washout,
)
from petrophys.preprocessing.quality.bad_hole.flag_generator import (
    bad_hole_flags,
    washout_flag,
    large_hole_flag,
    low_density_flag,
    pef_exclusion_flag,
)
from petrophys.preprocessing.quality.bad_hole.curve_masking import (
    mask_bad_hole_curves,
    apply_quality_flags,
)

__all__ = [
    "WashoutDetection",
    "detect_washout",
    "bad_hole_flags",
    "washout_flag",
    "large_hole_flag",
    "low_density_flag",
    "pef_exclusion_flag",
    "mask_bad_hole_curves",
    "apply_quality_flags",
]
