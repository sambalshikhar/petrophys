"""
Permeability from porosity and irreducible water saturation, or from core.

This module provides multiple permeability estimation methods
including Timur, Wyllie-Rose, Coates-Denoo, lith_vcl equations, and
impermeability detection.
"""

from petrophys.interpretation.permeability.perm_module import (
    CorePermModel,
    PermeabilityModule,
    calculate_permeability,
    estimate_apparent_swirr,
    estimate_hc_swirr,
    estimate_swirr_app,
    resolve_swirr,
)
from petrophys.interpretation.permeability.timur import timur
from petrophys.interpretation.permeability.coates_denoo import (
    WYLLIE_ROSE_PRESETS,
    coates_denoo,
    wyllie_rose,
)
from petrophys.interpretation.permeability.lith_vcl import lith_vcl_permeability
from petrophys.interpretation.permeability.classification import classify_permeability
from petrophys.interpretation.permeability.impermeable_flag import flag_impermeable

__all__ = [
    "CorePermModel",
    "PermeabilityModule",
    "calculate_permeability",
    "estimate_apparent_swirr",
    "estimate_hc_swirr",
    "estimate_swirr_app",
    "resolve_swirr",
    "timur",
    "coates_denoo",
    "wyllie_rose",
    "WYLLIE_ROSE_PRESETS",
    "lith_vcl_permeability",
    "classify_permeability",
    "flag_impermeable",
]
