"""Compatibility namespace for borehole-image interpretation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.interpretation.borehole_imaging",
    {
        "borehole_image": "petrophys.interpretation.borehole_imaging.borehole_image",
        "fmi": "petrophys.interpretation.borehole_imaging.fmi",
    },
)
