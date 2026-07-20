"""Compatibility namespace for clay-volume interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.clay_volume",
    {
        "classification": "petrophys.interpretation.clay_volume.classification",
        "gr_method": "petrophys.interpretation.clay_volume.gr_method",
        "nd_method": "petrophys.interpretation.clay_volume.nd_method",
        "rcl": "petrophys.interpretation.clay_volume.rcl",
        "sp_method": "petrophys.interpretation.clay_volume.sp_method",
        "vcl_module": "petrophys.interpretation.clay_volume.vcl_module",
    },
)
