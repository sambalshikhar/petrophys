"""
Reservoir evaluation: decisions and summaries built on interpreted curves.

Tools for:
- Bulk Volume Water (BVW)
- Moveable Hydrocarbon Index (MHI)
- Net Pay determination and pay summaries
- Rock Quality (RQI, FZI, phi_z)
- Sensitivity of net pay to cutoffs, parameters and saturation models
"""

from petrophys.evaluation.bvw import calculate_bvw, BvwModule
from petrophys.evaluation.mhi import calculate_mhi, MhiModule
from petrophys.evaluation.net_pay import (
    STANDARD_CUTOFFS,
    standard_cutoffs,
    calculate_net_pay,
    NetPayModule,
    NetPayResult,
    PaySummary,
    summarize_pay,
)
from petrophys.evaluation.rock_quality import (
    RockQualityModule,
    RockQualityResult,
    calculate_rock_quality,
    calculate_rqi,
    calculate_phi_z,
    calculate_fzi,
)
from petrophys.evaluation.sensitivity import (
    SATURATION_MODELS,
    SaturationModelComparison,
    compare_saturation_models,
    cutoff_sensitivity,
    parameter_sensitivity,
)
from petrophys.evaluation.flow_units import (
    FlowUnitModule,
    FlowUnitResult,
    calculate_flow_units,
    assign_flow_units,
    predict_perm_from_fzi,
)

__all__ = [
    "calculate_bvw",
    "BvwModule",
    "calculate_mhi",
    "MhiModule",
    "calculate_net_pay",
    "STANDARD_CUTOFFS",
    "standard_cutoffs",
    "NetPayModule",
    "NetPayResult",
    "PaySummary",
    "summarize_pay",
    "SATURATION_MODELS",
    "SaturationModelComparison",
    "compare_saturation_models",
    "cutoff_sensitivity",
    "parameter_sensitivity",
    "RockQualityModule",
    "RockQualityResult",
    "calculate_rock_quality",
    "calculate_rqi",
    "calculate_phi_z",
    "calculate_fzi",
    "FlowUnitModule",
    "FlowUnitResult",
    "calculate_flow_units",
    "assign_flow_units",
    "predict_perm_from_fzi",
]
