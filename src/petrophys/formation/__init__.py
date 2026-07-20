"""Compatibility namespace for interval models and evaluation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.evaluation.intervals",
    {
        "analysis": "petrophys.evaluation.intervals.analyzer",
        "intervals": "petrophys.model.intervals",
        "tops": "petrophys.model.formation_tops",
    },
)
