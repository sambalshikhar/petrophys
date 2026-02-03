"""
Lithology Determination Module.

This module provides tools for multi-mineral analysis and
lithology determination from well logs.
"""

from petrophys.formation.lithology.lithology_module import (
    LithologyModule,
    calculate_lithology,
)
from petrophys.formation.lithology.multi_mineral_solver import (
    MultiMineralSolver,
    solve_two_mineral,
    solve_three_mineral,
)
from petrophys.formation.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)
from petrophys.formation.lithology.crossplots import (
    nd_crossplot_data,
    mn_crossplot_data,
    pef_crossplot_data,
)

__all__ = [
    "LithologyModule",
    "calculate_lithology",
    "MultiMineralSolver",
    "solve_two_mineral",
    "solve_three_mineral",
    "get_mineral_properties",
    "MINERAL_PROPERTIES",
    "nd_crossplot_data",
    "mn_crossplot_data",
    "pef_crossplot_data",
]
