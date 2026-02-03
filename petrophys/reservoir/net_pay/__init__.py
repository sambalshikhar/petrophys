"""
Net Pay / Net Reservoir / Net Sand Module.
"""

from petrophys.reservoir.net_pay.net_pay_module import (
    NetPayModule,
    NetPayResult,
    calculate_net_pay,
)
from petrophys.reservoir.net_pay.cutoffs import Cutoffs, default_cutoffs
from petrophys.reservoir.net_pay.flags import generate_net_flags

__all__ = [
    "NetPayModule",
    "NetPayResult",
    "calculate_net_pay",
    "Cutoffs",
    "default_cutoffs",
    "generate_net_flags",
]
