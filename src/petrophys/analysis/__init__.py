"""Compatibility namespace for evaluation diagnostics."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.evaluation.diagnostics",
    {
        "hydrocarbon_analysis": "petrophys.evaluation.diagnostics.hydrocarbon_analysis",
        "lithology_analysis": "petrophys.evaluation.diagnostics.lithology_analysis",
        "numerical_statistical_analysis": "petrophys.evaluation.diagnostics.numerical_statistical_analysis",
        "permeability_analysis": "petrophys.evaluation.diagnostics.permeability_analysis",
        "porosity_analysis": "petrophys.evaluation.diagnostics.porosity_analysis",
        "quality_check": "petrophys.preprocessing.quality.wellbore",
        "saturation_analysis": "petrophys.evaluation.diagnostics.saturation_analysis",
    },
)

from petrophys.preprocessing.quality.wellbore import WellboreQC

__all__.append("WellboreQC")
