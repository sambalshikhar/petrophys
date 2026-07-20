"""Compatibility namespace for reservoir evaluation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.evaluation",
    {
        "bvw": "petrophys.evaluation.bvw",
        "bvw.bvw_module": "petrophys.evaluation.bvw.bvw_module",
        "flow_units": "petrophys.evaluation.flow_units",
        "flow_units.flow_units_module": "petrophys.evaluation.flow_units.flow_units_module",
        "mhi": "petrophys.evaluation.mhi",
        "mhi.classification": "petrophys.evaluation.mhi.classification",
        "mhi.mhi_module": "petrophys.evaluation.mhi.mhi_module",
        "net_pay": "petrophys.evaluation.net_pay",
        "net_pay.cutoffs": "petrophys.evaluation.net_pay.cutoffs",
        "net_pay.flags": "petrophys.evaluation.net_pay.flags",
        "net_pay.net_pay_module": "petrophys.evaluation.net_pay.net_pay_module",
        "rock_quality": "petrophys.evaluation.rock_quality",
        "rock_quality.rock_quality_module": "petrophys.evaluation.rock_quality.rock_quality_module",
    },
)
