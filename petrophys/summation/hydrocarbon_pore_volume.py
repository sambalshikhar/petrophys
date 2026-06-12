"""
Hydrocarbon pore volume and reserves calculations.
"""

from typing import Optional, Dict
from dataclasses import dataclass
import numpy as np

from petrophys.utils.analysis_helpers import (
    depth_step_array,
    resolve_well_depth_range,
    well_depth_array,
)


@dataclass
class HCPVResult:
    """Hydrocarbon pore volume calculation result."""

    # Input parameters
    area_acres: float
    net_pay_ft: float
    avg_porosity: float
    avg_sw: float

    # Results
    hcpv_acft: float      # Hydrocarbon pore volume in acre-feet
    hcpv_bbl: float       # Hydrocarbon pore volume in barrels
    hcpv_m3: float        # Hydrocarbon pore volume in cubic meters

    # For oil
    ooip_bbl: float       # Original Oil In Place (barrels)
    ooip_stb: float       # Original Oil In Place (stock tank barrels)

    # For gas
    ogip_mcf: float       # Original Gas In Place (thousand cubic feet)
    ogip_bcf: float       # Original Gas In Place (billion cubic feet)


def calculate_hcpv(
    area_acres: float,
    net_pay_ft: float,
    porosity: float,
    sw: float,
    bo: float = 1.2,
    bg: float = 0.005,
    ngi: float = 1.0
) -> HCPVResult:
    """
    Calculate hydrocarbon pore volume and reserves.

    Parameters
    ----------
    area_acres : float
        Drainage area in acres
    net_pay_ft : float
        Net pay thickness in feet
    porosity : float
        Average porosity (fraction)
    sw : float
        Average water saturation (fraction)
    bo : float
        Oil formation volume factor (rb/stb)
    bg : float
        Gas formation volume factor (rcf/scf)
    ngi : float
        Net-to-gross improvement factor

    Returns
    -------
    HCPVResult
        HCPV calculation results

    Notes
    -----
    Conversion factors:
    - 1 acre-foot = 7,758.4 barrels
    - 1 acre = 43,560 sq ft
    - 1 barrel = 0.159 m³
    """
    # Hydrocarbon saturation
    sh = 1.0 - sw

    # Bulk volume (acre-feet)
    bulk_volume_acft = area_acres * net_pay_ft

    # Pore volume (acre-feet)
    pore_volume_acft = bulk_volume_acft * porosity

    # Hydrocarbon pore volume (acre-feet)
    hcpv_acft = pore_volume_acft * sh * ngi

    # Convert to barrels (7758.4 bbl per acre-foot)
    hcpv_bbl = hcpv_acft * 7758.4

    # Convert to cubic meters
    hcpv_m3 = hcpv_bbl * 0.159

    # Oil in place (surface conditions)
    ooip_stb = hcpv_bbl / bo

    # Gas in place (surface conditions)
    # 1 acre-foot = 43,560 cubic feet
    hcpv_cf = hcpv_acft * 43560
    ogip_scf = hcpv_cf / bg
    ogip_mcf = ogip_scf / 1000
    ogip_bcf = ogip_scf / 1e9

    return HCPVResult(
        area_acres=area_acres,
        net_pay_ft=net_pay_ft,
        avg_porosity=porosity,
        avg_sw=sw,
        hcpv_acft=hcpv_acft,
        hcpv_bbl=hcpv_bbl,
        hcpv_m3=hcpv_m3,
        ooip_bbl=hcpv_bbl,
        ooip_stb=ooip_stb,
        ogip_mcf=ogip_mcf,
        ogip_bcf=ogip_bcf
    )


def calculate_ooip(
    area_acres: float,
    net_pay_ft: float,
    porosity: float,
    sw: float,
    bo: float = 1.2,
    recovery_factor: float = 0.3
) -> Dict[str, float]:
    """
    Calculate Original Oil In Place.

    Parameters
    ----------
    area_acres : float
        Drainage area in acres
    net_pay_ft : float
        Net pay thickness in feet
    porosity : float
        Average porosity (fraction)
    sw : float
        Average water saturation (fraction)
    bo : float
        Oil formation volume factor
    recovery_factor : float
        Expected recovery factor (fraction)

    Returns
    -------
    dict
        OOIP results with recoverable reserves

    Notes
    -----
    OOIP (STB) = 7758.4 × A × h × φ × (1-Sw) / Bo
    """
    sh = 1.0 - sw

    # OOIP in stock tank barrels
    ooip_stb = 7758.4 * area_acres * net_pay_ft * porosity * sh / bo

    # Recoverable reserves
    recoverable_stb = ooip_stb * recovery_factor

    # Convert to metric
    ooip_m3 = ooip_stb * 0.159

    return {
        "ooip_stb": ooip_stb,
        "ooip_mbbl": ooip_stb / 1000,
        "ooip_mmbbl": ooip_stb / 1e6,
        "ooip_m3": ooip_m3,
        "recoverable_stb": recoverable_stb,
        "recoverable_mbbl": recoverable_stb / 1000,
        "recovery_factor": recovery_factor,
    }


def calculate_ogip(
    area_acres: float,
    net_pay_ft: float,
    porosity: float,
    sw: float,
    bg: float = 0.005,
    recovery_factor: float = 0.7
) -> Dict[str, float]:
    """
    Calculate Original Gas In Place.

    Parameters
    ----------
    area_acres : float
        Drainage area in acres
    net_pay_ft : float
        Net pay thickness in feet
    porosity : float
        Average porosity (fraction)
    sw : float
        Average water saturation (fraction)
    bg : float
        Gas formation volume factor (rcf/scf)
    recovery_factor : float
        Expected recovery factor (fraction)

    Returns
    -------
    dict
        OGIP results with recoverable reserves

    Notes
    -----
    OGIP (scf) = 43560 × A × h × φ × (1-Sw) / Bg
    """
    sh = 1.0 - sw

    # OGIP in standard cubic feet
    ogip_scf = 43560 * area_acres * net_pay_ft * porosity * sh / bg

    # Recoverable reserves
    recoverable_scf = ogip_scf * recovery_factor

    return {
        "ogip_scf": ogip_scf,
        "ogip_mcf": ogip_scf / 1000,
        "ogip_mmcf": ogip_scf / 1e6,
        "ogip_bcf": ogip_scf / 1e9,
        "ogip_tcf": ogip_scf / 1e12,
        "recoverable_scf": recoverable_scf,
        "recoverable_bcf": recoverable_scf / 1e9,
        "recovery_factor": recovery_factor,
    }


def calculate_hcpv_from_well(
    well: "Well",
    area_acres: float,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    bo: float = 1.2,
    bg: float = 0.005
) -> HCPVResult:
    """
    Calculate HCPV using well log data.

    Parameters
    ----------
    well : Well
        Well object with computed curves
    area_acres : float
        Drainage area in acres
    top_depth : float, optional
        Top depth
    bottom_depth : float, optional
        Bottom depth
    bo : float
        Oil formation volume factor
    bg : float
        Gas formation volume factor

    Returns
    -------
    HCPVResult
        HCPV results
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

    # Get pay mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)

    net_pay = well.get("NET_PAY")
    if net_pay is not None:
        pay_mask = mask & (net_pay.data > 0.5)
    else:
        pay_mask = mask

    # Calculate net pay thickness
    net_pay_ft = float(np.sum(depth_step_array(depth)[pay_mask]))
    if net_pay_ft <= 0.0:
        return calculate_hcpv(
            area_acres=area_acres,
            net_pay_ft=0.0,
            porosity=0.0,
            sw=0.0,
            bo=bo,
            bg=bg
        )

    # Get average properties over pay
    phie = well.get("PHIE")
    sw = well.get("SW")

    if phie is not None:
        phie_data = phie.data[pay_mask]
        avg_porosity = float(np.nanmean(phie_data)) if np.any(np.isfinite(phie_data)) else np.nan
    else:
        avg_porosity = 0.15  # Default

    if sw is not None:
        sw_data = sw.data[pay_mask]
        avg_sw = float(np.nanmean(sw_data)) if np.any(np.isfinite(sw_data)) else np.nan
    else:
        avg_sw = 0.3  # Default

    return calculate_hcpv(
        area_acres=area_acres,
        net_pay_ft=net_pay_ft,
        porosity=avg_porosity,
        sw=avg_sw,
        bo=bo,
        bg=bg
    )
