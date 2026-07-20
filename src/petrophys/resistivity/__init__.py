"""Compatibility namespace for electrical interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.electrical",
    {
        "invasion": "petrophys.interpretation.electrical.invasion",
        "invasion.invasion_classification": "petrophys.interpretation.electrical.invasion.invasion_classification",
        "invasion.invasion_module": "petrophys.interpretation.electrical.invasion.invasion_module",
        "rmf": "petrophys.interpretation.electrical.rmf",
        "rmf.rmf_module": "petrophys.interpretation.electrical.rmf.rmf_module",
        "rw": "petrophys.interpretation.electrical.rw",
        "rw.rw_module": "petrophys.interpretation.electrical.rw.rw_module",
        "rw.salinity_rw": "petrophys.interpretation.electrical.rw.salinity_rw",
        "rw.sp_rw": "petrophys.interpretation.electrical.rw.sp_rw",
        "rw.temp_correction": "petrophys.interpretation.electrical.rw.temp_correction",
    },
)
