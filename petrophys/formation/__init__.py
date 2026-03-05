"""
Formation Module - Interval-based analysis.

This module provides tools for formation-level (interval-based) analysis,
such as loading formation tops from CSV and per-formation summaries.

Note: Whole-well calculations (clay volume, lithology) are top-level modules:
  - petrophys.clay_volume
  - petrophys.lithology
"""

# Backward-compatible re-exports (deprecated paths)
from petrophys.clay_volume import (
    calculate_vcl,
    vcl_from_gr,
    vcl_from_sp,
    vcl_from_nd,
    VclModule,
)
from petrophys.lithology import (
    MultiMineralSolver,
    calculate_lithology,
)

__all__ = [
    # Backward-compatible re-exports
    "calculate_vcl",
    "vcl_from_gr",
    "vcl_from_sp",
    "vcl_from_nd",
    "VclModule",
    "MultiMineralSolver",
    "calculate_lithology",
]
