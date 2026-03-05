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
from petrophys.petrophysics.saturation.archie import archie, archie_sw
from petrophys.petrophysics.saturation.simandoux import simandoux, simandoux_sw
from petrophys.petrophysics.saturation.indonesia import indonesia, indonesia_sw
from petrophys.petrophysics.saturation.dual_water import dual_water, dual_water_sw
from petrophys.petrophysics.saturation.waxman_smits import waxman_smits, waxman_smits_sw
from petrophys.petrophysics.saturation.sxo import calculate_sxo
from petrophys.petrophysics.saturation.classification import classify_sw

__all__ = [
    "SaturationModule",
    "SaturationResult",
    "SwModel",
    "calculate_sw",
    "archie",
    "archie_sw",
    "simandoux",
    "simandoux_sw",
    "indonesia",
    "indonesia_sw",
    "dual_water",
    "dual_water_sw",
    "waxman_smits",
    "waxman_smits_sw",
    "calculate_sxo",
    "classify_sw",
]
