"""Compatibility namespace for lithology interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.lithology",
    {
        "crossplots": "petrophys.interpretation.lithology.crossplots",
        "lithology_log": "petrophys.interpretation.lithology.lithology_log",
        "lithology_module": "petrophys.interpretation.lithology.lithology_module",
        "matrix_inference": "petrophys.interpretation.lithology.matrix_inference",
        "matrix_properties": "petrophys.interpretation.lithology.matrix_properties",
        "multi_mineral_solver": "petrophys.interpretation.lithology.multi_mineral_solver",
    },
)
