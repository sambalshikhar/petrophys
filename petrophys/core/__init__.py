"""
Core Sample Data Module.

Tools for loading, managing, and calibrating core laboratory measurements
against well log data:
- CoreData: Container for core plug measurements (porosity, perm, GD, Sw)
- Calibration: Core-vs-log comparison, poro-perm transform fitting, Archie m estimation
- Plots: Calibration crossplots, poro-perm plots, core-on-log overlays
"""

from petrophys.core.core_data import CoreData
from petrophys.core.calibration import (
    CalibrationResult,
    PoroPermTransform,
    calibrate_porosity,
    calibrate_permeability,
    fit_poro_perm_transform,
    apply_poro_perm_transform,
    correct_porosity_with_grain_density,
    estimate_archie_m,
    interpolate_log_at_core,
)
from petrophys.core.plots import (
    plot_calibration,
    plot_poro_perm,
    plot_core_on_logs,
)

__all__ = [
    # Data container
    "CoreData",
    # Calibration
    "CalibrationResult",
    "PoroPermTransform",
    "calibrate_porosity",
    "calibrate_permeability",
    "fit_poro_perm_transform",
    "apply_poro_perm_transform",
    "correct_porosity_with_grain_density",
    "estimate_archie_m",
    "interpolate_log_at_core",
    # Plots
    "plot_calibration",
    "plot_poro_perm",
    "plot_core_on_logs",
]
