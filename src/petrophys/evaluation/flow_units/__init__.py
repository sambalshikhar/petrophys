"""
Hydraulic Flow Units (Amaefule FZI method), and flow roles of zones from their permeability.

Determines discrete flow units from the FZI distribution and predicts
permeability from each unit's mean FZI.
"""

from petrophys.evaluation.flow_units.flow_units_module import (
    FlowUnitModule,
    FlowUnitResult,
    calculate_flow_units,
    assign_flow_units,
    predict_perm_from_fzi,
)
from petrophys.evaluation.flow_units.roles import flow_roles

__all__ = [
    "FlowUnitModule",
    "FlowUnitResult",
    "calculate_flow_units",
    "assign_flow_units",
    "predict_perm_from_fzi",
    "flow_roles",
]
