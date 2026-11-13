"""
Capillary pressure, saturation-height models and formation-pressure free-water levels.

Capillary-pressure primitives (Pc<->height, lab<->reservoir, Leverett J) plus
fitted saturation functions (J power-law, Brooks-Corey) and a log-column
Sw-vs-height predictor.
"""

from petrophys.interpretation.saturation.height.capillary import (
    pc_from_height,
    height_from_pc,
    convert_pc_lab_to_reservoir,
    leverett_j,
    pc_from_j,
    FLUID_SYSTEMS,
    WATER_GRADIENT_PSI_PER_FT,
    J_CONST,
)
from petrophys.interpretation.saturation.height.saturation_height_module import (
    JFunctionModel,
    BrooksCoreyModel,
    fit_j_function,
    fit_j_function_from_well,
    fit_brooks_corey,
    saturation_height_from_logs,
)
from petrophys.interpretation.saturation.height.formation_pressure import (
    PressureGradient, PressureFWLResult, fit_pressure_fwl,
    FormationPressureData, read_formation_pressure,
)
from petrophys.interpretation.saturation.height.pressure_contacts import (
    PressureBranch, PressureContact, PressureContactResult, infer_pressure_contacts,
)
from petrophys.interpretation.saturation.height.mdt_pretest import (
    MDTPickingConfig, MDTBuildup, MDTPressurePick, MDTPretest,
    read_mdt_pretest, pick_mdt_pressure, formation_pressures_from_mdt,
)

from petrophys.interpretation.saturation.height.scal import (
    CapillaryPressureData,
    FWLFromLogs,
    estimate_fwl_from_logs,
    fit_j_function_from_core,
    fit_thomeer,
    read_capillary_pressure,
    water_contact_from_fwl,
)

from petrophys.interpretation.saturation.height.empirical import (
    SATURATION_HEIGHT_MODELS,
    FittedSaturationHeight,
    fit_saturation_height,
    fit_saturation_height_from_well,
    height_above_fwl,
)

__all__ = [
    "SATURATION_HEIGHT_MODELS",
    "FittedSaturationHeight",
    "fit_saturation_height",
    "fit_saturation_height_from_well",
    "height_above_fwl",
    "pc_from_height",
    "height_from_pc",
    "convert_pc_lab_to_reservoir",
    "leverett_j",
    "pc_from_j",
    "FLUID_SYSTEMS",
    "WATER_GRADIENT_PSI_PER_FT",
    "J_CONST",
    "JFunctionModel",
    "BrooksCoreyModel",
    "fit_j_function",
    "fit_j_function_from_well",
    "CapillaryPressureData",
    "FWLFromLogs",
    "estimate_fwl_from_logs",
    "fit_j_function_from_core",
    "fit_thomeer",
    "read_capillary_pressure",
    "water_contact_from_fwl",
    "fit_brooks_corey",
    "saturation_height_from_logs",
    "PressureGradient",
    "PressureFWLResult",
    "fit_pressure_fwl",
    "FormationPressureData",
    "read_formation_pressure",
    "PressureBranch",
    "PressureContact",
    "PressureContactResult",
    "infer_pressure_contacts",
    "MDTPickingConfig",
    "MDTBuildup",
    "MDTPressurePick",
    "MDTPretest",
    "read_mdt_pretest",
    "pick_mdt_pressure",
    "formation_pressures_from_mdt",
]
