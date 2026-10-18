"""
Net-to-gross calculations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Union
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys._internal.analysis_helpers import (
    cell_thickness,
    resolve_well_depth_range,
    well_depth_array,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class NetGrossResult:
    """Net-to-gross calculation result."""

    gross_thickness: float
    net_rock_thickness: float
    net_reservoir_thickness: float
    net_pay_thickness: float

    ntg_rock: float      # Net rock / Gross
    ntg_reservoir: float  # Net reservoir / Gross
    ntg_pay: float       # Net pay / Gross
    ntg_res_rock: float  # Net reservoir / Net rock
    ntg_pay_res: float   # Net pay / Net reservoir


def calculate_net_gross(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: str = "NET_PAY",
    net_reservoir_curve: str = "NET_RES",
    net_rock_curve: str = "NET_ROCK",
    include_bottom: bool = True,
) -> NetGrossResult:
    """
    Calculate net-to-gross ratios.

    A net thickness whose flag curve is not on the well is NaN (unknown, not
    zero), and so is a ratio with a zero or unknown denominator.

    Parameters
    ----------
    well : Well
        Well object with net flags
    top_depth : float, optional
        Top depth (default: well start)
    bottom_depth : float, optional
        Bottom depth (default: well stop)
    net_pay_curve : str
        Name of net pay flag curve
    net_reservoir_curve : str
        Name of net reservoir flag curve
    net_rock_curve : str
        Name of net rock flag curve
    include_bottom : bool
        Count a sample at ``bottom_depth`` (False for a zone whose bottom is
        the next zone's top)

    Returns
    -------
    NetGrossResult
        Net-to-gross results
    """
    depth = well_depth_array(well)
    # Set depth range
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

    # Get depth mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    if not include_bottom:
        mask &= depth < bottom_depth
    sample_thickness = cell_thickness(depth)
    gross_thickness = float(np.sum(sample_thickness[mask]))
    if gross_thickness <= 0 and np.any(mask):
        gross_thickness = abs(bottom_depth - top_depth)

    # Get flags
    net_pay_c = well.get(net_pay_curve)
    net_res_c = well.get(net_reservoir_curve)
    net_rock_c = well.get(net_rock_curve)

    missing = [
        name for name, curve in ((net_rock_curve, net_rock_c), (net_reservoir_curve, net_res_c),
                                 (net_pay_curve, net_pay_c)) if curve is None
    ]
    if missing:
        warnings.warn(
            f"Net flag curve(s) {', '.join(missing)} not on the well; their thicknesses and ratios are NaN",
            UserWarning,
            stacklevel=2,
        )
    net_rock_thick = np.nan
    net_res_thick = np.nan
    net_pay_thick = np.nan
    zone_thickness = sample_thickness[mask]

    if net_rock_c is not None:
        net_rock_thick = float(np.sum(zone_thickness[net_rock_c.data[mask] > 0.5]))

    if net_res_c is not None:
        net_res_thick = float(np.sum(zone_thickness[net_res_c.data[mask] > 0.5]))

    if net_pay_c is not None:
        net_pay_thick = float(np.sum(zone_thickness[net_pay_c.data[mask] > 0.5]))

    # Calculate ratios
    ntg_rock = net_rock_thick / gross_thickness if gross_thickness > 0 else np.nan
    ntg_reservoir = net_res_thick / gross_thickness if gross_thickness > 0 else np.nan
    ntg_pay = net_pay_thick / gross_thickness if gross_thickness > 0 else np.nan
    ntg_res_rock = net_res_thick / net_rock_thick if net_rock_thick > 0 else np.nan
    ntg_pay_res = net_pay_thick / net_res_thick if net_res_thick > 0 else np.nan

    return NetGrossResult(
        gross_thickness=gross_thickness,
        net_rock_thickness=net_rock_thick,
        net_reservoir_thickness=net_res_thick,
        net_pay_thickness=net_pay_thick,
        ntg_rock=ntg_rock,
        ntg_reservoir=ntg_reservoir,
        ntg_pay=ntg_pay,
        ntg_res_rock=ntg_res_rock,
        ntg_pay_res=ntg_pay_res
    )


def calculate_ntg_by_zone(
    well: "Well",
    zones: List[Dict[str, Union[str, float]]]
) -> Dict[str, NetGrossResult]:
    """
    Calculate net-to-gross for multiple zones.

    Parameters
    ----------
    well : Well
        Well object
    zones : list
        List of zone definitions with 'name', 'top', 'bottom'

    Returns
    -------
    dict
        Dictionary mapping zone names to NetGrossResult
    """
    results = {}
    tops = {float(zone["top"]) for zone in zones}

    for zone in zones:
        name = zone.get("name", f"Zone_{zone['top']}")
        result = calculate_net_gross(
            well,
            top_depth=zone["top"],
            bottom_depth=zone["bottom"],
            # A boundary sample shared with the next zone is counted once, in that zone.
            include_bottom=float(zone["bottom"]) not in tops,
        )
        results[name] = result

    return results


def create_ntg_table(
    well: "Well",
    zones: List[Dict[str, Union[str, float]]]
) -> str:
    """
    Create a formatted net-to-gross table.

    Parameters
    ----------
    well : Well
        Well object
    zones : list
        List of zone definitions

    Returns
    -------
    str
        Formatted table string
    """
    results = calculate_ntg_by_zone(well, zones)

    # Header
    lines = [
        "=" * 80,
        f"{'Zone':<15} {'Gross':>8} {'N.Rock':>8} {'N.Res':>8} {'N.Pay':>8} {'NTG':>8}",
        "-" * 80,
    ]

    # Rows
    for name, result in results.items():
        lines.append(
            f"{name:<15} {result.gross_thickness:>8.2f} "
            f"{result.net_rock_thickness:>8.2f} {result.net_reservoir_thickness:>8.2f} "
            f"{result.net_pay_thickness:>8.2f} {result.ntg_pay:>8.1%}"
        )

    lines.append("=" * 80)

    return "\n".join(lines)
