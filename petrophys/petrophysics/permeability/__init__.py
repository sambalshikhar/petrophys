"""
Permeability Estimation Module.

This module provides multiple permeability estimation methods
including Timur, Wyllie-Rose, Coates-Denoo equations, and
impermeability detection.
"""

from petrophys.petrophysics.permeability.perm_module import (
    CorePermModel,
    PermeabilityModule,
    calculate_permeability,
    estimate_apparent_swirr,
    resolve_swirr,
)
from petrophys.petrophysics.permeability.timur import timur, timur_perm
from petrophys.petrophysics.permeability.coates_denoo import coates_denoo
from petrophys.petrophysics.permeability.classification import classify_permeability
from petrophys.petrophysics.permeability.impermeable_flag import flag_impermeable

# Backward-compatible alias
PermModule = PermeabilityModule

__all__ = [
    "CorePermModel",
    "PermeabilityModule",
    "PermModule",
    "calculate_permeability",
    "estimate_apparent_swirr",
    "resolve_swirr",
    "timur",
    "timur_perm",
    "coates_denoo",
    "classify_permeability",
    "flag_impermeable",
]
