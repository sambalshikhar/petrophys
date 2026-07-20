"""
Lithology Determination Module.

This module provides tools for multi-mineral analysis and
lithology determination from well logs, as well as interval-based
lithology assignment from CSV files.
"""

from petrophys.interpretation.lithology.lithology_module import (
    LithologyModule,
    calculate_lithology,
)
from petrophys.interpretation.lithology.multi_mineral_solver import (
    MultiMineralSolver,
    solve_two_mineral,
    solve_three_mineral,
)
from petrophys.interpretation.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)
from petrophys.interpretation.lithology.crossplots import (
    nd_crossplot_data,
    mn_crossplot_data,
    pef_crossplot_data,
)
from petrophys.interpretation.lithology.lithology_log import (
    LithologyInterval,
    load_lithology_csv,
    build_lithology_log,
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
    "LithologyInterval",
    "load_lithology_csv",
    "build_lithology_log",
]
