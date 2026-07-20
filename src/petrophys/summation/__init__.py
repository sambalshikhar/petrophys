"""Compatibility namespace for evaluation summaries."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.evaluation.summaries",
    {
        "averages": "petrophys.evaluation.summaries.averages",
        "formation_analysis": "petrophys.evaluation.summaries.formation_analysis",
        "hydrocarbon_pore_volume": "petrophys.evaluation.volumetrics.hydrocarbon_pore_volume",
        "net_gross": "petrophys.evaluation.summaries.net_gross",
        "zone_summary": "petrophys.evaluation.summaries.zone_summary",
    },
)
