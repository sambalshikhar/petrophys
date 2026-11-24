"""Core calibration and digital-core interpretation.

The package owns laboratory calibration, core-to-log depth matching,
rock-mechanics measurements, grain size analysis, digital facies, and
core-image measurements. Plotting helpers
live in :mod:`petrophys.visualization.core`.
"""

from petrophys.interpretation.core_analysis.calibration import (
    CalibrationResult,
    PoroPermTransform,
    PorosityCorrection,
    apply_poro_perm,
    apply_porosity_correction,
    calibrate_permeability,
    calibrate_porosity,
    calibrate_saturation,
    core_data,
    correct_porosity_with_grain_density,
    estimate_archie_m,
    fit_log_poro_perm_transform,
    fit_log_poro_perm_transform_from_correction,
    fit_poro_perm,
    fit_porosity_correction,
    interpolate_log_at_core,
)
from petrophys.interpretation.core_analysis.color_analysis import CoreImageData
from petrophys.interpretation.core_analysis.depth_matching import (
    CoreDepthShift,
    find_core_depth_shift,
    shift_core_depths,
)
from petrophys.interpretation.core_analysis.digital_facies import (
    DigitalCoreFacies,
    FaciesResult,
)
from petrophys.interpretation.core_analysis.grain_size import (
    GrainSizeData,
    um_to_phi,
    wentworth_class,
)
from petrophys.interpretation.core_analysis.mechanical import CoreMechanical
from petrophys.model.core import CoreData

__all__ = [
    "CoreData",
    "CalibrationResult",
    "PorosityCorrection",
    "PoroPermTransform",
    "calibrate_porosity",
    "calibrate_permeability",
    "calibrate_saturation",
    "fit_porosity_correction",
    "apply_porosity_correction",
    "fit_poro_perm",
    "fit_log_poro_perm_transform",
    "fit_log_poro_perm_transform_from_correction",
    "apply_poro_perm",
    "correct_porosity_with_grain_density",
    "estimate_archie_m",
    "interpolate_log_at_core",
    "core_data",
    "CoreDepthShift",
    "find_core_depth_shift",
    "shift_core_depths",
    "CoreMechanical",
    "GrainSizeData",
    "wentworth_class",
    "um_to_phi",
    "DigitalCoreFacies",
    "FaciesResult",
    "CoreImageData",
]
