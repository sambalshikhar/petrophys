"""
Permeability Estimation Module.

This module provides multiple permeability estimation methods
including Timur, Wyllie-Rose, Coates-Denoo equations, and
impermeability detection.
"""

from petrophys.petrophysics.permeability.perm_module import (
    PermeabilityModule,
    calculate_permeability,
)
from petrophys.petrophysics.permeability.timur import timur, timur_perm
from petrophys.petrophysics.permeability.coates_denoo import coates_denoo
from petrophys.petrophysics.permeability.classification import classify_permeability
from petrophys.petrophysics.permeability.impermeable_flag import flag_impermeable

__all__ = [
    "PermeabilityModule",
    "calculate_permeability",
    "timur",
    "timur_perm",
    "coates_denoo",
    "classify_permeability",
    "flag_impermeable",
]
