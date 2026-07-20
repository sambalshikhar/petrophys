"""
Porosity Module.

This module provides comprehensive porosity calculations including
density, neutron, sonic, total, and effective porosity.
"""

from petrophys.interpretation.porosity.porosity_module import (
    PorosityModule,
    calculate_porosity,
)
from petrophys.interpretation.porosity.density_porosity import (
    density_porosity,
    calculate_phid,
)
from petrophys.interpretation.porosity.neutron_porosity import (
    neutron_porosity,
    standardize_nphi,
)
from petrophys.interpretation.porosity.sonic_porosity import (
    sonic_porosity,
    wyllie_time_average,
)
from petrophys.interpretation.porosity.total_porosity import (
    total_porosity,
    nd_weighted_porosity,
)
from petrophys.interpretation.porosity.effective_porosity import (
    effective_porosity,
    calculate_phie,
)
from petrophys.interpretation.porosity.classification import classify_porosity

__all__ = [
    "PorosityModule",
    "calculate_porosity",
    "density_porosity",
    "calculate_phid",
    "neutron_porosity",
    "standardize_nphi",
    "sonic_porosity",
    "wyllie_time_average",
    "total_porosity",
    "nd_weighted_porosity",
    "effective_porosity",
    "calculate_phie",
    "classify_porosity",
]
