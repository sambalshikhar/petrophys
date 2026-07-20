"""Compatibility namespace for formation-temperature interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.temperature",
    {
        "temperature": "petrophys.interpretation.temperature",
        "temperature.temp_corrections": "petrophys.interpretation.temperature.temp_corrections",
        "temperature.temp_gradient": "petrophys.interpretation.temperature.temp_gradient",
        "temperature.temp_profile": "petrophys.interpretation.temperature.temp_profile",
    },
)
