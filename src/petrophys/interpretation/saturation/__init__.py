"""
Water saturation from Archie and shaly-sand models, and flushed-zone saturation.

This module provides multiple water saturation models including
Archie, Simandoux, Indonesia, Waxman-Smits, and Dual Water.
"""

from petrophys.interpretation.saturation.sw_module import (
    SaturationModule,
    SaturationResult,
    SwModel,
    calculate_sw,
)
from petrophys.interpretation.saturation.archie import archie
from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand
from petrophys.interpretation.saturation.nigeria_schlumberger import nigeria_schlumberger
from petrophys.interpretation.saturation.simandoux import simandoux
from petrophys.interpretation.saturation.indonesia import indonesia
from petrophys.interpretation.saturation.dual_water import dual_water
from petrophys.interpretation.saturation.waxman_smits import waxman_smits
from petrophys.interpretation.saturation.sxo import calculate_sxo
from petrophys.interpretation.saturation.classification import classify_sw

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
