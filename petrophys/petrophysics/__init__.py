"""
Petrophysics Umbrella Module - Core petrophysical calculations.

This umbrella module contains tools for:
- Porosity calculation (density, neutron, sonic, combined)
- Hydrocarbon correction for neutron and density
- Water saturation (Archie, Simandoux, Indonesia, Waxman-Smits, Dual Water)
- Permeability estimation
- Cementation/Archie parameters
"""

from petrophys.petrophysics.porosity import (
    calculate_porosity,
    PorosityModule,
    density_porosity,
    sonic_porosity,
    total_porosity,
    effective_porosity,
)
from petrophys.petrophysics.hc_correction import (
    HcCorrectionModule,
    correct_neutron_hc,
    correct_density_hc,
)
from petrophys.petrophysics.saturation import (
    calculate_sw,
    SaturationModule,
    archie,
    simandoux,
    indonesia,
)
from petrophys.petrophysics.permeability import (
    calculate_permeability,
    PermeabilityModule,
    timur,
    coates_denoo,
)
from petrophys.petrophysics.cementation import (
    ArchieParameters,
    pickett_analysis,
)

__all__ = [
    # Porosity
    "calculate_porosity",
    "PorosityModule",
    "density_porosity",
    "sonic_porosity",
    "total_porosity",
    "effective_porosity",
    # HC Correction
    "HcCorrectionModule",
    "correct_neutron_hc",
    "correct_density_hc",
    # Saturation
    "calculate_sw",
    "SaturationModule",
    "archie",
    "simandoux",
    "indonesia",
    # Permeability
    "calculate_permeability",
    "PermeabilityModule",
    "timur",
    "coates_denoo",
    # Cementation
    "ArchieParameters",
    "pickett_analysis",
]
