"""Compatibility namespace for :mod:`petrophys.model`."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.model",
    {
        "classifications": "petrophys.model.classifications",
        "constants": "petrophys.model.constants",
        "curve": "petrophys.model.curve",
        "data_types": "petrophys.model.data_types",
        "log_data": "petrophys.model.log_data",
        "units": "petrophys.model.units",
        "well": "petrophys.model.well",
    },
)
