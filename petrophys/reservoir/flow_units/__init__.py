"""
Hydraulic Flow Units (Amaefule FZI method).

Determines discrete flow units from the FZI distribution and predicts
permeability from each unit's mean FZI.
"""

from petrophys.reservoir.flow_units.flow_units_module import (
    FlowUnitModule,
    FlowUnitResult,
    calculate_flow_units,
    assign_flow_units,
    predict_perm_from_fzi,
)

__all__ = [
    "FlowUnitModule",
    "FlowUnitResult",
    "calculate_flow_units",
    "assign_flow_units",
    "predict_perm_from_fzi",
]
