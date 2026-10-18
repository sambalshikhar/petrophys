"""
Zone summary calculations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Union
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.analysis_helpers import (
    cell_thickness,
    get_curve_data_depthwise,
    well_depth_array,
)
from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well


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
    # NOTE: hcpv here is a hydrocarbon pore THICKNESS, the sum of
    # PHIE*(1-SW)*thickness over the net-pay samples (units of depth/length,
    # per unit area), NOT a volume. For volumetric OOIP/OGIP in acre-ft/bbl/m3
    # use volumetrics.hydrocarbon_pore_volume.calculate_hcpv. The benchmark
    # reproduction keeps the index avg_phie*(1-avg_sw)*net_pay.
    hcpv: float = np.nan  # Hydrocarbon pore thickness (length, per unit area)
    # Bulk volume water (fraction): thickness-weighted mean of PHIE*SW over
    # net pay (avg_phie*avg_sw in the benchmark reproduction)
    bvw: float = np.nan

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
    net_rock_curve: str = "NET_ROCK",
    include_bottom: bool = True,
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
    include_bottom : bool
        Count a sample at ``bottom_depth``

    Returns
    -------
    ZoneSummary
        Zone summary results; a net thickness (and NTG) whose flag curve is
        not on the well is NaN
    """
    depth = well_depth_array(well)
    # Get depth mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    if not include_bottom:
        mask &= depth < bottom_depth
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

    sample_thickness = cell_thickness(depth)
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

    if not reproducing_benchmark():
        # A missing flag curve is unknown net thickness, not zero.
        net_rock_thick = net_rock_thick if net_rock_c is not None else np.nan
        net_res_thick = net_res_thick if net_res_c is not None else np.nan
        net_pay_thick = net_pay_thick if net_pay_c is not None else np.nan

    ntg = net_pay_thick / gross_thickness if gross_thickness > 0 else 0

    # Calculate averages over net pay
    pay_mask = mask & (net_pay_c.data > 0.5) if net_pay_c is not None else np.zeros_like(mask, dtype=bool)
    n_pay_samples = int(np.sum(pay_mask))

    if reproducing_benchmark():
        avg_phie = _get_average(well, "PHIE", pay_mask)
        avg_sw = _get_average(well, "SW", pay_mask)
        avg_vcl = _get_average(well, "VCL", pay_mask)
        avg_perm = _get_geometric_average(well, "PERM", pay_mask)

        # Calculate HCPV and BVW
        hcpv = avg_phie * (1 - avg_sw) * net_pay_thick if not np.isnan(avg_phie) else np.nan
        bvw = avg_phie * avg_sw if not np.isnan(avg_phie) else np.nan
    else:
        # The interpretation's SW and NET_PAY come from the core-corrected
        # porosity where it exists, so the pay averages prefer it too.
        phie = _pay_data(well, ["PHIE_CC", "PHIE"])
        sw = _pay_data(well, ["SW"])
        avg_phie = _mean(phie[pay_mask])
        avg_sw = _mean(sw[pay_mask])
        avg_vcl = _get_average(well, "VCL", pay_mask)
        avg_perm = _geometric_mean(_pay_data(well, ["PERM_CC", "PERM"])[pay_mask])

        # HCPV sums PHIE*(1-SW)*thickness over pay; BVW is the thickness-weighted
        # mean of PHIE*SW over pay, so hcpv = (avg_phie - bvw)*net_pay when the
        # pay samples share one thickness.
        hc_valid = pay_mask & np.isfinite(phie) & np.isfinite(sw)
        if np.any(hc_valid):
            h = sample_thickness[hc_valid]
            hcpv = float(np.sum(phie[hc_valid] * (1.0 - sw[hc_valid]) * h))
            bvw = float(np.sum(phie[hc_valid] * sw[hc_valid] * h) / np.sum(h)) if np.sum(h) > 0 else np.nan
        else:
            hcpv = np.nan
            bvw = np.nan

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


def _pay_data(well: "Well", names: List[str]) -> np.ndarray:
    """Depth-wise composite of the first finite alias per sample (NaN without any)."""
    data, _, _ = get_curve_data_depthwise(well, names)
    if data is None:
        return np.full(well.n_samples, np.nan, dtype=float)
    return np.asarray(data, dtype=float)


def _mean(data: np.ndarray) -> float:
    valid = data[np.isfinite(data)]
    return float(np.mean(valid)) if len(valid) else np.nan


def _geometric_mean(data: np.ndarray) -> float:
    valid = data[np.isfinite(data) & (data > 0)]
    return float(np.exp(np.mean(np.log(valid)))) if len(valid) else np.nan


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
    tops = {float(zone["top"]) for zone in zones}

    for zone in zones:
        summary = calculate_zone_summary(
            well,
            top_depth=zone["top"],
            bottom_depth=zone["bottom"],
            zone_name=zone.get("name", "Zone"),
            # A boundary sample shared with the next zone is counted once, in that zone.
            include_bottom=float(zone["bottom"]) not in tops,
        )
        summaries.append(summary)

    return summaries
