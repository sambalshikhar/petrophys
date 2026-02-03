"""
Summation and reporting module for petrophysics results.
"""

from petrophys.summation.zone_summary import (
    ZoneSummary,
    calculate_zone_summary,
    summarize_zones,
)
from petrophys.summation.net_gross import (
    NetGrossResult,
    calculate_net_gross,
    calculate_ntg_by_zone,
)
from petrophys.summation.averages import (
    calculate_zone_averages,
    calculate_pay_averages,
    calculate_weighted_averages,
)
from petrophys.summation.hydrocarbon_pore_volume import (
    HCPVResult,
    calculate_hcpv,
    calculate_ooip,
    calculate_ogip,
)

__all__ = [
    # Zone summary
    "ZoneSummary",
    "calculate_zone_summary",
    "summarize_zones",
    # Net gross
    "NetGrossResult",
    "calculate_net_gross",
    "calculate_ntg_by_zone",
    # Averages
    "calculate_zone_averages",
    "calculate_pay_averages",
    "calculate_weighted_averages",
    # HCPV
    "HCPVResult",
    "calculate_hcpv",
    "calculate_ooip",
    "calculate_ogip",
]
