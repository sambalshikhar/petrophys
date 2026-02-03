"""
Formation Umbrella Module - Formation characterization.

This umbrella module contains tools for:
- Clay volume (VCL) calculation
- Lithology determination
"""

from petrophys.formation.clay_volume import (
    calculate_vcl,
    vcl_from_gr,
    vcl_from_sp,
    vcl_from_nd,
    VclModule,
)
from petrophys.formation.lithology import (
    MultiMineralSolver,
    calculate_lithology,
)

__all__ = [
    # Clay volume
    "calculate_vcl",
    "vcl_from_gr",
    "vcl_from_sp",
    "vcl_from_nd",
    "VclModule",
    # Lithology
    "MultiMineralSolver",
    "calculate_lithology",
]
