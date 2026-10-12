"""
Hydrocarbon pore volume and reserves calculations.
"""

from typing import TYPE_CHECKING, Optional, Dict
from dataclasses import dataclass
import numpy as np

from petrophys.model.units import normalize_depth_unit
from petrophys._internal.analysis_helpers import (
    cell_thickness,
    resolve_well_depth_range,
    well_depth_array,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


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
        # OOIP is a surface (stock-tank) quantity: divide reservoir HCPV by Bo.
        # Previously ooip_bbl was set to the reservoir-condition hcpv_bbl, which
        # mislabeled reservoir barrels as oil-in-place.
        ooip_bbl=ooip_stb,
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

    Net pay is the thickness of the samples flagged by ``NET_PAY`` (midpoint
    cells from the actual depths), converted from the well's depth unit to
    feet. Over net pay, porosity is thickness-weighted and water saturation
    pore-volume weighted (by PHIE * h), so that
    ``net_pay_ft * avg_porosity * (1 - avg_sw)`` is the sum of
    ``PHIE * (1 - SW) * h``, the hydrocarbon pore thickness.

    Parameters
    ----------
    well : Well
        Well object with ``NET_PAY``, ``PHIE`` and ``SW`` curves
    area_acres : float
        Drainage area in acres
    top_depth : float, optional
        Top depth, in the well's depth unit
    bottom_depth : float, optional
        Bottom depth, in the well's depth unit
    bo : float
        Oil formation volume factor
    bg : float
        Gas formation volume factor

    Returns
    -------
    HCPVResult
        HCPV results

    Raises
    ------
    ValueError
        If ``NET_PAY``, ``PHIE`` or ``SW`` is missing, or net pay has no
        finite PHIE and SW.
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)
    missing = [name for name in ("NET_PAY", "PHIE", "SW") if well.get(name) is None]
    if missing:
        raise ValueError(
            f"calculate_hcpv_from_well needs {', '.join(missing)} on the well; "
            "run the interpretation first"
        )

    # Get pay mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    pay_mask = mask & (np.asarray(well.get("NET_PAY").data, dtype=float) > 0.5)

    # Net pay thickness in feet for the field-unit volumes
    feet_per_unit = 1.0 / 0.3048 if normalize_depth_unit(well.depth_unit, strict=False) == "m" else 1.0
    thickness_ft = cell_thickness(depth) * feet_per_unit
    net_pay_ft = float(np.sum(thickness_ft[pay_mask]))
    if net_pay_ft <= 0.0:
        return calculate_hcpv(
            area_acres=area_acres,
            net_pay_ft=0.0,
            porosity=0.0,
            sw=0.0,
            bo=bo,
            bg=bg
        )

    # Thickness-weighted porosity and pore-volume-weighted Sw over pay
    phie = np.asarray(well.get("PHIE").data, dtype=float)
    sw = np.asarray(well.get("SW").data, dtype=float)
    valid = pay_mask & np.isfinite(phie) & np.isfinite(sw)
    if not np.any(valid):
        raise ValueError("Net pay has no samples with finite PHIE and SW")
    h = thickness_ft[valid]
    pore_ft = phie[valid] * h
    avg_porosity = float(np.sum(pore_ft) / np.sum(h))
    if np.sum(pore_ft) > 0.0:
        avg_sw = float(np.sum(sw[valid] * pore_ft) / np.sum(pore_ft))
    else:
        avg_sw = float(np.sum(sw[valid] * h) / np.sum(h))

    return calculate_hcpv(
        area_acres=area_acres,
        net_pay_ft=net_pay_ft,
        porosity=avg_porosity,
        sw=avg_sw,
        bo=bo,
        bg=bg
    )
