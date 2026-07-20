"""Scientific interpretation of well, core, and specialist-log data.

This namespace owns deterministic calculations that turn measurements into
petrophysical properties. Reservoir decisions and aggregations live in
``petrophys.evaluation`` instead.
"""

from petrophys.interpretation.clay_volume import VclModule, calculate_vcl
from petrophys.interpretation.electrical.cementation import (
    ArchieParameters,
    pickett_analysis,
)
from petrophys.interpretation.lithology import (
    MultiMineralSolver,
    calculate_lithology,
)
from petrophys.interpretation.permeability import (
    PermeabilityModule,
    calculate_permeability,
)
from petrophys.interpretation.porosity import PorosityModule, calculate_porosity
from petrophys.interpretation.saturation import SaturationModule, calculate_sw

__all__ = [
    "ArchieParameters",
    "MultiMineralSolver",
    "PermeabilityModule",
    "PorosityModule",
    "SaturationModule",
    "VclModule",
    "calculate_lithology",
    "calculate_permeability",
    "calculate_porosity",
    "calculate_sw",
    "calculate_vcl",
    "pickett_analysis",
]
