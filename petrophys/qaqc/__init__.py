"""
QAQC Umbrella Module - Quality Assurance / Quality Control.

This umbrella module contains tools for:
- Curve standardization (alias mapping, unit conversion)
- Bad hole detection (washout, flags)
- Log normalization (multi-well consistency)
"""

from petrophys.qaqc.standardization import (
    CurveStandardizer,
    standardize_curves,
    ALIAS_MAP,
)
from petrophys.qaqc.bad_hole import (
    WashoutDetection,
    detect_washout,
    generate_bad_hole_flags,
)
from petrophys.qaqc.normalization import (
    normalize_log,
    histogram_matching,
    NormalizationModule,
)

__all__ = [
    # Standardization
    "CurveStandardizer",
    "standardize_curves",
    "ALIAS_MAP",
    # Bad Hole
    "WashoutDetection",
    "detect_washout",
    "generate_bad_hole_flags",
    # Normalization
    "normalize_log",
    "histogram_matching",
    "NormalizationModule",
]
