"""
Reservoir Umbrella Module - Reservoir evaluation.

This umbrella module contains tools for:
- Bulk Volume Water (BVW)
- Moveable Hydrocarbon Index (MHI)
- Net Pay determination
- Rock Quality (RQI, FZI, phi_z)
"""

from petrophys.reservoir.bvw import calculate_bvw, BvwModule
from petrophys.reservoir.mhi import calculate_mhi, MhiModule
from petrophys.reservoir.net_pay import (
    calculate_net_pay,
    NetPayModule,
    NetPayResult,
)
from petrophys.reservoir.rock_quality import (
    RockQualityModule,
    RockQualityResult,
    calculate_rock_quality,
    calculate_rqi,
    calculate_phi_z,
    calculate_fzi,
)
from petrophys.reservoir.flow_units import (
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
    "NetPayModule",
    "NetPayResult",
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
