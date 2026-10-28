"""
Clay volume (VCL) from gamma ray, SP or neutron-density, and clay resistivity.

This module provides multiple methods for calculating clay volume
from well log data, plus clay resistivity (Rcl) estimation.
"""

from petrophys.interpretation.clay_volume.vcl_module import (
    VclModule,
    calculate_vcl,
)
from petrophys.interpretation.clay_volume.gr_method import (
    bounded_gr_baselines,
    vcl_from_gr,
    vcl_linear,
    vcl_larionov_tertiary,
    vcl_larionov_older,
    vcl_steiber,
    vcl_clavier,
)
from petrophys.interpretation.clay_volume.sp_method import vcl_from_sp
from petrophys.interpretation.clay_volume.nd_method import vcl_from_nd
from petrophys.interpretation.clay_volume.classification import classify_vcl
from petrophys.interpretation.clay_volume.rcl import calculate_rcl, RclModule
from petrophys.interpretation.clay_volume.thomas_stieber import (
    ThomasStieberResult,
    laminated_sand_resistivity,
    thomas_stieber,
)

__all__ = [
    "VclModule",
    "calculate_vcl",
    "bounded_gr_baselines",
    "vcl_from_gr",
    "vcl_linear",
    "vcl_larionov_tertiary",
    "vcl_larionov_older",
    "vcl_steiber",
    "vcl_clavier",
    "vcl_from_sp",
    "vcl_from_nd",
    "classify_vcl",
    "calculate_rcl",
    "RclModule",
    "ThomasStieberResult",
    "laminated_sand_resistivity",
    "thomas_stieber",
]
