"""
Formation Water Resistivity (Rw) Module.

This module provides multiple methods for calculating formation
water resistivity at formation temperature.
"""

from petrophys.interpretation.electrical.rw.rw_module import RwModule, calculate_rw
from petrophys.interpretation.electrical.rw.sp_rw import calculate_ssp, estimate_sp_lines, rw_from_sp, rw_from_ssp
from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity, salinity_from_rw
from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature

__all__ = [
    "RwModule",
    "calculate_rw",
    "estimate_sp_lines",
    "rw_from_sp",
    "rw_from_ssp",
    "calculate_ssp",
    "rw_from_salinity",
    "salinity_from_rw",
    "correct_rw_temperature",
]
