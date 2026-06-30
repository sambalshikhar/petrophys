"""
Water Saturation Module.

This module provides multiple water saturation models including
Archie, Simandoux, Indonesia, Waxman-Smits, and Dual Water.
"""

from petrophys.petrophysics.saturation.sw_module import (
    SaturationModule,
    SaturationResult,
    SwModel,
    calculate_sw,
)
from petrophys.petrophysics.saturation.archie import archie
from petrophys.petrophysics.saturation.north_sea import north_sea_shaly_sand
from petrophys.petrophysics.saturation.nigeria_schlumberger import nigeria_schlumberger
from petrophys.petrophysics.saturation.simandoux import simandoux
from petrophys.petrophysics.saturation.indonesia import indonesia
from petrophys.petrophysics.saturation.dual_water import dual_water
from petrophys.petrophysics.saturation.waxman_smits import waxman_smits
from petrophys.petrophysics.saturation.sxo import calculate_sxo
from petrophys.petrophysics.saturation.classification import classify_sw

__all__ = [
    "SaturationModule",
    "SaturationResult",
    "SwModel",
    "calculate_sw",
    "archie",
    "north_sea_shaly_sand",
    "nigeria_schlumberger",
    "simandoux",
    "indonesia",
    "dual_water",
    "waxman_smits",
    "calculate_sxo",
    "classify_sw",
]
