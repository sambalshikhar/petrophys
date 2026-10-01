"""
Invasion profiles from shallow and deep resistivity.

This module provides tools for characterizing invasion profiles
from shallow and deep resistivity measurements.
"""

from petrophys.interpretation.electrical.invasion.invasion_module import (
    InvasionModule,
    calculate_invasion_ratio,
)
from petrophys.interpretation.electrical.invasion.invasion_classification import (
    classify_invasion,
    InvasionType,
)

__all__ = [
    "InvasionModule",
    "calculate_invasion_ratio",
    "classify_invasion",
    "InvasionType",
]
