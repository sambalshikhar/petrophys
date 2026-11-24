"""
Net rock, net reservoir and net pay flags from cutoffs, and pay summaries.
"""

from petrophys.evaluation.net_pay.net_pay_module import (
    NetPayModule,
    NetPayResult,
    calculate_net_pay,
)
from petrophys.evaluation.net_pay.cutoffs import (
    Cutoffs, STANDARD_CUTOFFS, default_cutoffs, standard_cutoffs,
)
from petrophys.evaluation.net_pay.pay_summary import PaySummary, summarize_pay
from petrophys.evaluation.net_pay.screening import (
    cutoff_attrition,
    grade_interval,
    lithology_in,
    majority_label,
)
from petrophys.evaluation.net_pay.flags import (
    DEFAULT_HYDROCARBON_FLUIDS,
    DEFAULT_NON_PAY_FLUIDS,
    add_volumetric_pay_flag,
    net_flags,
    volumetric_pay_flag,
)

__all__ = [
    "NetPayModule",
    "NetPayResult",
    "calculate_net_pay",
    "PaySummary",
    "summarize_pay",
    "Cutoffs",
    "default_cutoffs",
    "STANDARD_CUTOFFS",
    "standard_cutoffs",
    "net_flags",
    "volumetric_pay_flag",
    "add_volumetric_pay_flag",
    "DEFAULT_HYDROCARBON_FLUIDS",
    "DEFAULT_NON_PAY_FLUIDS",
    "grade_interval",
    "cutoff_attrition",
    "lithology_in",
    "majority_label",
]
