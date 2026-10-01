"""
Hydrocarbon corrections for neutron and density logs.

This module provides corrections for neutron and density logs
affected by hydrocarbon presence.
"""

from petrophys.interpretation.porosity.hydrocarbon_correction.hc_correction_module import (
    HcCorrectionModule,
    apply_hc_correction,
)
from petrophys.interpretation.porosity.hydrocarbon_correction.neutron_correction import correct_neutron_hc
from petrophys.interpretation.porosity.hydrocarbon_correction.density_correction import correct_density_hc

__all__ = [
    "HcCorrectionModule",
    "apply_hc_correction",
    "correct_neutron_hc",
    "correct_density_hc",
]
