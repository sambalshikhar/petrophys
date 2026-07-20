"""
Zone summary calculations.
"""

from typing import Optional, List, Dict, Union
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.analysis_helpers import depth_step_array, well_depth_array


@dataclass
class ZoneSummary:
    """Summary results for a zone or interval."""

    zone_name: str
    top_depth: float
    bottom_depth: float
    gross_thickness: float
    net_rock: float
    net_reservoir: float
    net_pay: float
    ntg: float  # Net-to-gross (pay/gross)

    # Average properties (over net pay)
    avg_phie: float = np.nan
    avg_sw: float = np.nan
    avg_vcl: float = np.nan
    avg_perm: float = np.nan

    # Hydrocarbon properties
    # NOTE: hcpv here is a hydrocarbon-column INDEX = avg_phie*(1-avg_sw)*net_pay
    # (units of depth/length, per unit area), NOT a volume. For volumetric
    # OOIP/OGIP in acre-ft/bbl/m3 use summation.hydrocarbon_pore_volume.calculate_hcpv.
    hcpv: float = np.nan  # Hydrocarbon pore thickness index (length, per unit area)
    bvw: float = np.nan   # Bulk volume water (fraction): avg_phie*avg_sw

    # Statistics
    n_samples: int = 0
    n_pay_samples: int = 0


def calculate_zone_summary(
    well: "Well",
    top_depth: float,
    bottom_depth: float,
    zone_name: str = "Zone",
    net_pay_curve: str = "NET_PAY",
    net_reservoir_curve: str = "NET_RES",
    net_rock_curve: str = "NET_ROCK"
) -> ZoneSummary:
    """
    Calculate summary for a depth interval.

    Parameters
    ----------
    well : Well
        Well object with computed curves
    top_depth : float
        Top depth of zone
    bottom_depth : float
        Bottom depth of zone
    zone_name : str
        Name of the zone
    net_pay_curve : str
        Name of net pay flag curve
    net_reservoir_curve : str
        Name of net reservoir flag curve
    net_rock_curve : str
        Name of net rock flag curve

    Returns
    -------
    ZoneSummary
        Zone summary results
    """
    depth = well_depth_array(well)
    # Get depth mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    n_samples = np.sum(mask)

    if n_samples == 0:
        return ZoneSummary(
            zone_name=zone_name,
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            gross_thickness=0,
            net_rock=0,
            net_reservoir=0,
            net_pay=0,
            ntg=0,
            n_samples=0,
            n_pay_samples=0
        )

    sample_thickness = depth_step_array(depth)
    gross_thickness = float(np.sum(sample_thickness[mask]))
    if gross_thickness <= 0:
        gross_thickness = abs(bottom_depth - top_depth)

    # Get flags
    net_pay_c = well.get(net_pay_curve)
    net_res_c = well.get(net_reservoir_curve)
    net_rock_c = well.get(net_rock_curve)

    net_pay_flag = net_pay_c.data[mask] if net_pay_c is not None else np.zeros(n_samples)
    net_res_flag = net_res_c.data[mask] if net_res_c is not None else np.zeros(n_samples)
    net_rock_flag = net_rock_c.data[mask] if net_rock_c is not None else np.zeros(n_samples)

    zone_thickness = sample_thickness[mask]
    net_rock_thick = float(np.sum(zone_thickness[net_rock_flag > 0.5]))
    net_res_thick = float(np.sum(zone_thickness[net_res_flag > 0.5]))
    net_pay_thick = float(np.sum(zone_thickness[net_pay_flag > 0.5]))

    ntg = net_pay_thick / gross_thickness if gross_thickness > 0 else 0

    # Calculate averages over net pay
    pay_mask = mask & (net_pay_c.data > 0.5) if net_pay_c is not None else np.zeros_like(mask, dtype=bool)
    n_pay_samples = int(np.sum(pay_mask))

    avg_phie = _get_average(well, "PHIE", pay_mask)
    avg_sw = _get_average(well, "SW", pay_mask)
    avg_vcl = _get_average(well, "VCL", pay_mask)
    avg_perm = _get_geometric_average(well, "PERM", pay_mask)

    # Calculate HCPV and BVW
    hcpv = avg_phie * (1 - avg_sw) * net_pay_thick if not np.isnan(avg_phie) else np.nan
    bvw = avg_phie * avg_sw if not np.isnan(avg_phie) else np.nan

    return ZoneSummary(
        zone_name=zone_name,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        gross_thickness=gross_thickness,
        net_rock=net_rock_thick,
        net_reservoir=net_res_thick,
        net_pay=net_pay_thick,
        ntg=ntg,
        avg_phie=avg_phie,
        avg_sw=avg_sw,
        avg_vcl=avg_vcl,
        avg_perm=avg_perm,
        hcpv=hcpv,
        bvw=bvw,
        n_samples=n_samples,
        n_pay_samples=n_pay_samples
    )


def _get_average(well: "Well", curve_name: str, mask: np.ndarray) -> float:
    """Get average value over mask."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan

    data = curve.data[mask]
    valid = data[~np.isnan(data)]

    if len(valid) == 0:
        return np.nan

    return float(np.mean(valid))


def _get_geometric_average(well: "Well", curve_name: str, mask: np.ndarray) -> float:
    """Get geometric average (for permeability)."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan

    data = curve.data[mask]
    valid = data[(~np.isnan(data)) & (data > 0)]

    if len(valid) == 0:
        return np.nan

    return float(np.exp(np.mean(np.log(valid))))


def summarize_zones(
    well: "Well",
    zones: List[Dict[str, Union[str, float]]]
) -> List[ZoneSummary]:
    """
    Calculate summaries for multiple zones.

    Parameters
    ----------
    well : Well
        Well object
    zones : list
        List of zone definitions, each with:
        - name: str
        - top: float
        - bottom: float

    Returns
    -------
    list
        List of ZoneSummary objects
    """
    summaries = []

    for zone in zones:
        summary = calculate_zone_summary(
            well,
            top_depth=zone["top"],
            bottom_depth=zone["bottom"],
            zone_name=zone.get("name", "Zone")
        )
        summaries.append(summary)

    return summaries
