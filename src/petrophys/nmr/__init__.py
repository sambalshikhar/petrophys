"""Compatibility namespace for NMR interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.nmr",
    {
        "calibration": "petrophys.interpretation.nmr.calibration",
        "nmr_data": "petrophys.interpretation.nmr.nmr_data",
        "permeability": "petrophys.interpretation.nmr.permeability",
    },
)
