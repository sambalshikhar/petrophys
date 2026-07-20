"""Compatibility namespace for preprocessing and quality control."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.preprocessing",
    {
        "bad_hole": "petrophys.preprocessing.quality.bad_hole",
        "normalization": "petrophys.preprocessing.normalization",
        "standardization": "petrophys.preprocessing.standardization",
        "value_range_qc": "petrophys.preprocessing.quality.value_ranges",
    },
)
