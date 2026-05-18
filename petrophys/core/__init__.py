"""
Core Sample Data Module.

Tools for loading, managing, and calibrating core laboratory measurements
against well log data:

- CoreData: Container for core plug measurements (porosity, perm, GD, Sw)
- Calibration: Core-vs-log comparison, poro-perm transform fitting, Archie m estimation
- Plots: Calibration crossplots, poro-perm plots, core-on-log overlays
- XRFData: XRF geochemistry data (elemental concentrations, mineral proxies)
- CoreMechanical: UCS, Vp, Vs, dynamic elastic moduli from core scanning
- GrainSizeData: Grain size distributions, Wentworth classification, perm estimates
- DigitalCoreFacies / FaciesResult: Weighted K-means facies from multi-parameter data
- CoreImageData: RGB/HSV color analysis from core photographs
"""

from petrophys.core.core_data import CoreData
from petrophys.core.calibration import (
    CalibrationResult,
    PorosityCorrection,
    PoroPermTransform,
    calibrate_porosity,
    calibrate_permeability,
    fit_porosity_correction,
    apply_porosity_correction,
    fit_poro_perm_transform,
    fit_log_poro_perm_transform,
    fit_log_poro_perm_transform_from_correction,
    apply_poro_perm_transform,
    correct_porosity_with_grain_density,
    estimate_archie_m,
    interpolate_log_at_core,
    core_data_from_well,
)
from petrophys.core.plots import (
    plot_calibration,
    plot_poro_perm,
    plot_core_on_logs,
)
from petrophys.core.xrf import XRFData
from petrophys.core.mechanical import CoreMechanical
from petrophys.core.grain_size import GrainSizeData, wentworth_class, um_to_phi
from petrophys.core.digital_facies import DigitalCoreFacies, FaciesResult
from petrophys.core.color_analysis import CoreImageData

__all__ = [
    # Routine core data
    "CoreData",
    # Calibration
    "CalibrationResult",
    "PorosityCorrection",
    "PoroPermTransform",
    "calibrate_porosity",
    "calibrate_permeability",
    "fit_porosity_correction",
    "apply_porosity_correction",
    "fit_poro_perm_transform",
    "fit_log_poro_perm_transform",
    "fit_log_poro_perm_transform_from_correction",
    "apply_poro_perm_transform",
    "correct_porosity_with_grain_density",
    "estimate_archie_m",
    "interpolate_log_at_core",
    "core_data_from_well",
    # Plots
    "plot_calibration",
    "plot_poro_perm",
    "plot_core_on_logs",
    # Digital core — geochemistry
    "XRFData",
    # Digital core — mechanical
    "CoreMechanical",
    # Digital core — grain size
    "GrainSizeData",
    "wentworth_class",
    "um_to_phi",
    # Digital core — facies
    "DigitalCoreFacies",
    "FaciesResult",
    # Digital core — color / image
    "CoreImageData",
]
