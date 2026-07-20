"""
Summation and reporting module for petrophysics results.
"""

from petrophys.evaluation.summaries.zone_summary import (
    ZoneSummary,
    calculate_zone_summary,
    summarize_zones,
)
from petrophys.evaluation.summaries.net_gross import (
    NetGrossResult,
    calculate_net_gross,
    calculate_ntg_by_zone,
)
from petrophys.evaluation.summaries.averages import (
    calculate_zone_averages,
    calculate_pay_averages,
    calculate_weighted_averages,
)
from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import (
    HCPVResult,
    calculate_hcpv,
    calculate_ooip,
    calculate_ogip,
)
from petrophys.evaluation.summaries.formation_analysis import (
    Formation,
    FormationSummary,
    FormationAnalysisResult,
    load_formations_csv,
    analyze_formations,
    run_formation_analysis,
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
    # Formation analysis
    "Formation",
    "FormationSummary",
    "FormationAnalysisResult",
    "load_formations_csv",
    "analyze_formations",
    "run_formation_analysis",
]
