"""
Formation Water Resistivity (Rw) Module.

This module provides multiple methods for calculating formation
water resistivity at formation temperature.
"""

from petrophys.resistivity.rw.rw_module import RwModule, calculate_rw
from petrophys.resistivity.rw.sp_rw import rw_from_sp, calculate_ssp
from petrophys.resistivity.rw.salinity_rw import rw_from_salinity, salinity_from_rw
from petrophys.resistivity.rw.temp_correction import correct_rw_temperature

__all__ = [
    "RwModule",
    "calculate_rw",
    "rw_from_sp",
    "calculate_ssp",
    "rw_from_salinity",
    "salinity_from_rw",
    "correct_rw_temperature",
]
