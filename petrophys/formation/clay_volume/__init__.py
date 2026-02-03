"""
Clay Volume (VCL) Module.

This module provides multiple methods for calculating clay volume
from well log data.
"""

from petrophys.formation.clay_volume.vcl_module import (
    VclModule,
    calculate_vcl,
)
from petrophys.formation.clay_volume.gr_method import (
    vcl_from_gr,
    vcl_linear,
    vcl_larionov_tertiary,
    vcl_larionov_older,
    vcl_steiber,
    vcl_clavier,
)
from petrophys.formation.clay_volume.sp_method import vcl_from_sp
from petrophys.formation.clay_volume.nd_method import vcl_from_nd
from petrophys.formation.clay_volume.classification import classify_vcl

__all__ = [
    "VclModule",
    "calculate_vcl",
    "vcl_from_gr",
    "vcl_linear",
    "vcl_larionov_tertiary",
    "vcl_larionov_older",
    "vcl_steiber",
    "vcl_clavier",
    "vcl_from_sp",
    "vcl_from_nd",
    "classify_vcl",
]
