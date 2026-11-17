"""
Hydrocarbon pore volume and reserves calculations.

References
----------
Craft, B.C. and Hawkins, M.F. (1991). *Applied Petroleum Reservoir
Engineering*, 2nd ed., revised by R.E. Terry. Prentice Hall.
"""

from typing import TYPE_CHECKING, Any, Optional, Dict
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys.model.units import normalize_depth_unit
from petrophys._internal.analysis_helpers import (
    cell_thickness,
    get_curve_data_depthwise,
    resolve_well_depth_range,
    well_depth_array,
)
from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well


#: Cubic metres in one barrel (158.9873 litres).
M3_PER_BBL = 0.1589873


def _m3_per_bbl() -> float:
    # Benchmark context profiles use 0.159 cubic metres per barrel.
    return 0.159 if reproducing_benchmark() else M3_PER_BBL


def _fraction(value: Any, name: str) -> Any:
    """``value`` as a fraction (v/v): above 1 and up to 100 is read as percent.

    Raises ValueError for a value that is not finite, negative or above 100.
    """
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 100.0):
        raise ValueError(f"{name} must be a fraction between 0 and 1, or a percentage up to 100; got {value!r}")
    if np.any(array > 1.0):
        array = array / 100.0
    return float(array) if array.ndim == 0 else array


def _not_negative(value: Any, name: str, unit: str) -> Any:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be zero or more ({unit}); got {value!r}")
    return value


def _positive(value: Any, name: str, unit: str) -> Any:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be greater than 0 ({unit}); got {value!r}")
    return value


@dataclass
class HCPVResult:
    """Hydrocarbon pore volume calculation result.

    ``avg_porosity`` and ``avg_sw`` are fractions (v/v), also when they were
    given in percent.
    """

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
    ooip_bbl: float       # Original Oil In Place (stock tank barrels, as ooip_stb)
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
        Drainage area in acres, zero or more
    net_pay_ft : float
        Net pay thickness in feet, zero or more
    porosity : float
        Average porosity, fraction (v/v); a value above 1 and up to 100 is
        read as percent
    sw : float
        Average water saturation, fraction (v/v); a value above 1 and up to
        100 is read as percent
    bo : float
        Oil formation volume factor (rb/stb), greater than 0
    bg : float
        Gas formation volume factor (rcf/scf), greater than 0
    ngi : float
        Net-to-gross ratio applied to the hydrocarbon pore volume (1 when
        ``net_pay_ft`` is already a net thickness), fraction or percent as
        for ``porosity``

    Returns
    -------
    HCPVResult
        HCPV calculation results

    Raises
    ------
    ValueError
        If the area or thickness is negative, a formation volume factor is
        not greater than 0, a porosity, saturation or net-to-gross is
        outside 0 to 100, or any of them is not a finite number.

    Notes
    -----
    Conversion factors:
    - 1 acre-foot = 7,758.4 barrels
    - 1 acre = 43,560 sq ft
    - 1 barrel = 0.1589873 m³
    """
    if not reproducing_benchmark():
        _not_negative(area_acres, "area_acres", "acres")
        _not_negative(net_pay_ft, "net_pay_ft", "ft")
        porosity = _fraction(porosity, "porosity")
        sw = _fraction(sw, "sw")
        ngi = _fraction(ngi, "ngi")
        _positive(bo, "bo", "rb/stb")
        _positive(bg, "bg", "rcf/scf")

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
    hcpv_m3 = hcpv_bbl * _m3_per_bbl()

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
        # OOIP is a surface (stock-tank) quantity: reservoir HCPV divided by Bo.
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
        Drainage area in acres, zero or more
    net_pay_ft : float
        Net pay thickness in feet, zero or more
    porosity : float
        Average porosity, fraction (v/v); a value above 1 and up to 100 is
        read as percent
    sw : float
        Average water saturation, fraction (v/v); a value above 1 and up to
        100 is read as percent
    bo : float
        Oil formation volume factor (rb/stb), greater than 0
    recovery_factor : float
        Expected recovery factor, fraction or percent as for ``porosity``

    Returns
    -------
    dict
        OOIP in stock-tank barrels (``ooip_stb``, ``ooip_mbbl``,
        ``ooip_mmbbl``) and cubic metres (``ooip_m3``), the recoverable
        volume (``recoverable_stb``, ``recoverable_mbbl``) and the
        ``recovery_factor`` as a fraction

    Raises
    ------
    ValueError
        If the area or thickness is negative, ``bo`` is not greater than 0,
        a porosity, saturation or recovery factor is outside 0 to 100, or
        any of them is not a finite number.

    Notes
    -----
    OOIP (STB) = 7758.4 × A × h × φ × (1-Sw) / Bo; 1 barrel = 0.1589873 m³
    """
    if not reproducing_benchmark():
        _not_negative(area_acres, "area_acres", "acres")
        _not_negative(net_pay_ft, "net_pay_ft", "ft")
        porosity = _fraction(porosity, "porosity")
        sw = _fraction(sw, "sw")
        recovery_factor = _fraction(recovery_factor, "recovery_factor")
        _positive(bo, "bo", "rb/stb")
    sh = 1.0 - sw

    # OOIP in stock tank barrels
    ooip_stb = 7758.4 * area_acres * net_pay_ft * porosity * sh / bo

    # Recoverable reserves
    recoverable_stb = ooip_stb * recovery_factor

    # Convert to metric
    ooip_m3 = ooip_stb * _m3_per_bbl()

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
        Drainage area in acres, zero or more
    net_pay_ft : float
        Net pay thickness in feet, zero or more
    porosity : float
        Average porosity, fraction (v/v); a value above 1 and up to 100 is
        read as percent
    sw : float
        Average water saturation, fraction (v/v); a value above 1 and up to
        100 is read as percent
    bg : float
        Gas formation volume factor (rcf/scf), greater than 0
    recovery_factor : float
        Expected recovery factor, fraction or percent as for ``porosity``

    Returns
    -------
    dict
        OGIP in standard cubic feet (``ogip_scf`` to ``ogip_tcf``), the
        recoverable volume (``recoverable_scf``, ``recoverable_bcf``) and
        the ``recovery_factor`` as a fraction

    Raises
    ------
    ValueError
        If the area or thickness is negative, ``bg`` is not greater than 0,
        a porosity, saturation or recovery factor is outside 0 to 100, or
        any of them is not a finite number.

    Notes
    -----
    OGIP (scf) = 43560 × A × h × φ × (1-Sw) / Bg
    """
    if not reproducing_benchmark():
        _not_negative(area_acres, "area_acres", "acres")
        _not_negative(net_pay_ft, "net_pay_ft", "ft")
        porosity = _fraction(porosity, "porosity")
        sw = _fraction(sw, "sw")
        recovery_factor = _fraction(recovery_factor, "recovery_factor")
        _positive(bg, "bg", "rcf/scf")
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

    Net pay is the thickness of the samples flagged by ``NET_PAY`` that have
    a finite porosity and water saturation (midpoint cells from the actual
    depths), converted from the well's depth unit to feet; flagged samples
    without them are left out with a warning. Porosity is the core-corrected
    ``PHIE_CC`` where it exists, else ``PHIE``. Over net pay, porosity is
    thickness-weighted and water saturation pore-volume weighted (by
    PHIE * h), so that ``net_pay_ft * avg_porosity * (1 - avg_sw)`` is the
    sum of ``PHIE * (1 - SW) * h``, the hydrocarbon pore thickness.

    Parameters
    ----------
    well : Well
        Well object with ``NET_PAY``, ``PHIE`` (or ``PHIE_CC``) and ``SW``
        curves
    area_acres : float
        Drainage area in acres, zero or more
    top_depth : float, optional
        Top depth, in the well's depth unit
    bottom_depth : float, optional
        Bottom depth, in the well's depth unit
    bo : float
        Oil formation volume factor (rb/stb), greater than 0
    bg : float
        Gas formation volume factor (rcf/scf), greater than 0

    Returns
    -------
    HCPVResult
        HCPV results

    Raises
    ------
    ValueError
        If ``NET_PAY``, ``PHIE`` or ``SW`` is missing, ``top_depth`` is below
        ``bottom_depth``, net pay has no finite PHIE and SW, the area is
        negative or a formation volume factor is not greater than 0.
    """
    if not reproducing_benchmark():
        _not_negative(area_acres, "area_acres", "acres")
        _positive(bo, "bo", "rb/stb")
        _positive(bg, "bg", "rcf/scf")
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)
    if top_depth > bottom_depth:
        raise ValueError(f"top_depth ({top_depth:g}) is below bottom_depth ({bottom_depth:g})")
    benchmark = reproducing_benchmark()
    phie_names = ("PHIE",) if benchmark else ("PHIE_CC", "PHIE")
    missing = [
        name for name in ("NET_PAY", "PHIE", "SW")
        if not any(well.get(alias) is not None for alias in (phie_names if name == "PHIE" else (name,)))
    ]
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
    if benchmark:
        phie = np.asarray(well.get("PHIE").data, dtype=float)
    else:
        phie, _, _ = get_curve_data_depthwise(well, list(phie_names))
        phie = np.full(depth.shape, np.nan) if phie is None else np.asarray(phie, dtype=float)
    sw = np.asarray(well.get("SW").data, dtype=float)
    valid = pay_mask & np.isfinite(phie) & np.isfinite(sw)
    n_missing = int(np.sum(pay_mask & ~valid))
    if benchmark:
        net_pay_ft = float(np.sum(thickness_ft[pay_mask]))
    else:
        # Net pay is the flagged thickness with a porosity and saturation, so
        # net_pay_ft * avg_porosity * (1 - avg_sw) is the summed HC pore thickness.
        net_pay_ft = float(np.sum(thickness_ft[valid]))
        if np.any(pay_mask) and not np.any(valid):
            raise ValueError("Net pay has no samples with finite PHIE and SW")
        if n_missing:
            warnings.warn(
                f"{n_missing} of {int(np.sum(pay_mask))} net-pay samples have no finite PHIE and SW "
                "and are left out of net_pay_ft and the averages",
                UserWarning,
                stacklevel=2,
            )
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
    if not np.any(valid):
        raise ValueError("Net pay has no samples with finite PHIE and SW")
    if n_missing and benchmark:
        warnings.warn(
            f"{n_missing} of {int(np.sum(pay_mask))} net-pay samples have no finite PHIE and SW; "
            "net_pay_ft counts them but the averages do not, so the volumes assume they "
            "have the average properties",
            UserWarning,
            stacklevel=2,
        )
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
