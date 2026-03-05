"""
Rock Quality Module.

Provides Rock Quality Index (RQI), Flow Zone Indicator (FZI),
and normalized porosity (phi_z) calculations.
"""

from petrophys.reservoir.rock_quality.rock_quality_module import (
    RockQualityModule,
    RockQualityResult,
    calculate_rock_quality,
    calculate_rqi,
    calculate_phi_z,
    calculate_fzi,
)

__all__ = [
    "RockQualityModule",
    "RockQualityResult",
    "calculate_rock_quality",
    "calculate_rqi",
    "calculate_phi_z",
    "calculate_fzi",
]
