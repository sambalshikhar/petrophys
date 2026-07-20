"""Compatibility namespace for core models and interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.core_analysis",
    {
        "calibration": "petrophys.interpretation.core_analysis.calibration",
        "color_analysis": "petrophys.interpretation.core_analysis.color_analysis",
        "core_data": "petrophys.model.core",
        "digital_facies": "petrophys.interpretation.core_analysis.digital_facies",
        "grain_size": "petrophys.interpretation.core_analysis.grain_size",
        "mechanical": "petrophys.interpretation.core_analysis.mechanical",
        "plots": "petrophys.visualization.core",
        "xrf": "petrophys.interpretation.geochemistry.xrf",
    },
)

from petrophys.interpretation.geochemistry.xrf import XRFData
from petrophys.model.core import CoreData
from petrophys.visualization.core import (
    plot_calibration,
    plot_core_on_logs,
    plot_poro_perm,
)

__all__ += [
    "CoreData",
    "XRFData",
    "plot_calibration",
    "plot_core_on_logs",
    "plot_poro_perm",
]
