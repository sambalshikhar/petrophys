"""
Rock quality index (RQI), flow zone indicator (FZI) and normalized porosity.

Provides Rock Quality Index (RQI), Flow Zone Indicator (FZI),
and normalized porosity (phi_z) calculations.
"""

from petrophys.evaluation.rock_quality.rock_quality_module import (
    RockQualityModule,
    RockQualityResult,
    calculate_rock_quality,
    calculate_rqi,
    calculate_phi_z,
    calculate_fzi,
    winland_r35,
    classify_pore_throat,
    PORE_THROAT_CLASSES,
)

__all__ = [
    "RockQualityModule",
    "RockQualityResult",
    "calculate_rock_quality",
    "calculate_rqi",
    "calculate_phi_z",
    "calculate_fzi",
    "winland_r35",
    "classify_pore_throat",
    "PORE_THROAT_CLASSES",
]
