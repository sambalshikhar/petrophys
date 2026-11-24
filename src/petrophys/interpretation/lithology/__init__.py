"""
Mineral volumes, matrix properties, lithology crossplots and lithology logs.

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
    MultiMineralSolverV2,
    solve_two_mineral,
    solve_three_mineral,
)
from petrophys.interpretation.lithology.multi_mineral_solver_v1 import (
    MultiMineralSolverV1,
)
from petrophys.interpretation.lithology.matrix_properties import (
    mineral_properties,
    MINERAL_PROPERTIES,
)
from petrophys.interpretation.lithology.crossplots import (
    nd_crossplot_data,
    mn_crossplot_data,
    pef_crossplot_data,
)
from petrophys.interpretation.lithology.apparent_matrix import (
    apparent_matrix_density,
    apparent_matrix_slowness,
    apparent_matrix_volumetric_pe,
    apparent_total_porosity,
    nd_crossplot_porosity,
    electron_density,
    volumetric_photoelectric,
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
    "MultiMineralSolverV1",
    "MultiMineralSolverV2",
    "solve_two_mineral",
    "solve_three_mineral",
    "mineral_properties",
    "MINERAL_PROPERTIES",
    "nd_crossplot_data",
    "mn_crossplot_data",
    "pef_crossplot_data",
    "LithologyInterval",
    "load_lithology_csv",
    "build_lithology_log",
    "apparent_matrix_density",
    "apparent_matrix_slowness",
    "apparent_matrix_volumetric_pe",
    "apparent_total_porosity",
    "nd_crossplot_porosity",
    "electron_density",
    "volumetric_photoelectric",
]
