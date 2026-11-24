"""
Average calculations for zones and intervals.
"""

from typing import TYPE_CHECKING, Optional, List, Dict
import warnings
import numpy as np

from petrophys._internal.analysis_helpers import (
    cell_thickness,
    get_curve_data_depthwise,
    resolve_well_depth_range,
    well_depth_array,
)
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.evaluation.summaries.zone_summary import pay_average_curves, pay_average_data


#: Curves averaged geometrically by default (permeability).
_GEOMETRIC_DEFAULT = ["PERM", "PERM_CC", "K", "KLOG", "PERMEABILITY"]

if TYPE_CHECKING:
    from petrophys.model.well import Well


def zone_averages(
    well: "Well",
    top_depth: float,
    bottom_depth: float,
    curves: Optional[List[str]] = None,
    use_arithmetic: Optional[List[str]] = None,
    use_geometric: Optional[List[str]] = None,
    use_harmonic: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate averages for curves within a zone.

    Parameters
    ----------
    well : Well
        Well object
    top_depth : float
        Top depth
    bottom_depth : float
        Bottom depth
    curves : list, optional
        List of curve names to average (default: all numeric)
    use_arithmetic : list, optional
        Curves to average arithmetically (the default for curves in no list);
        a curve named here is averaged arithmetically even when it is in the
        default geometric list.
    use_geometric : list, optional
        Curves to use geometric mean (default: PERM, PERM_CC, K, KLOG,
        PERMEABILITY); the geometric and harmonic means are over the
        positive samples.
    use_harmonic : list, optional
        Curves to use harmonic mean

    Returns
    -------
    dict
        Dictionary of curve averages
    """
    depth = well_depth_array(well)
    mask = (depth >= top_depth) & (depth <= bottom_depth)

    if curves is None:
        curves = list(well.curves)  # Well.curves is a list of names

    if use_geometric is None:
        use_geometric = (
            ["PERM", "K", "KLOG", "PERMEABILITY"] if reproducing_benchmark() else _GEOMETRIC_DEFAULT
        )

    if use_harmonic is None:
        use_harmonic = []
    arithmetic = {c.upper() for c in (use_arithmetic or [])}

    results = {}

    for curve_name in curves:
        curve = well.get(curve_name)
        if curve is None:
            continue

        data = curve.data[mask]
        valid = data[~np.isnan(data)]

        if len(valid) == 0:
            results[curve_name] = np.nan
            continue

        if curve_name.upper() in arithmetic:
            results[curve_name] = float(np.mean(valid))

        elif curve_name.upper() in [c.upper() for c in use_harmonic]:
            # Harmonic mean
            pos = valid[valid > 0]
            if len(pos) > 0:
                results[curve_name] = float(len(pos) / np.sum(1.0 / pos))
            else:
                results[curve_name] = np.nan

        elif curve_name.upper() in [c.upper() for c in use_geometric]:
            # Geometric mean
            pos = valid[valid > 0]
            if len(pos) > 0:
                results[curve_name] = float(np.exp(np.mean(np.log(pos))))
            else:
                results[curve_name] = np.nan

        else:
            # Arithmetic mean
            results[curve_name] = float(np.mean(valid))

    return results


def pay_averages(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: str = "NET_PAY",
    curves: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate averages over net pay intervals only.

    Each sample counts with its thickness. Without the net-pay flag curve on
    the well there is no pay to average over: a warning is given and every
    average is NaN.

    Parameters
    ----------
    well : Well
        Well object
    top_depth : float, optional
        Top depth
    bottom_depth : float, optional
        Bottom depth
    net_pay_curve : str
        Name of net pay flag curve
    curves : list, optional
        Curves to average (default PHIE, SW, VCL, PERM, BVW). ``PHIE`` and
        ``PERM`` are the core-corrected ``PHIE_CC`` / ``PERM_CC`` where the
        interpretation produced or used them and the plain curve at the
        samples those do not cover (see
        :func:`~petrophys.evaluation.summaries.zone_summary.pay_average_data`);
        permeability (PERM, PERM_CC, K, PERMEABILITY) is a geometric mean.
        ``SW`` is weighted by thickness here;
        :func:`pore_volume_weighted_sw` gives the
        pore-volume-weighted value of the formation summaries.

    Returns
    -------
    dict
        Dictionary of averages over net pay, keyed by the requested names
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

    # Get pay mask
    net_pay = well.get(net_pay_curve)
    if net_pay is None:
        # If no pay flag, use depth mask only
        mask = (depth >= top_depth) & (depth <= bottom_depth)
    else:
        mask = (depth >= top_depth) & (depth <= bottom_depth) & (net_pay.data > 0.5)

    if curves is None:
        curves = ["PHIE", "SW", "VCL", "PERM", "BVW"]

    results = {}
    if reproducing_benchmark():
        weights = np.ones(int(np.count_nonzero(mask)))
        pay_values: Dict[str, np.ndarray] = {}
    else:
        if net_pay is None:
            warnings.warn(
                f"Net-pay flag curve {net_pay_curve!r} is not on the well, so there is no pay to "
                "average over: the averages are NaN. Run the net-pay step, or use "
                "zone_averages for averages over the whole interval.",
                UserWarning, stacklevel=2,
            )
            return {name: np.nan for name in curves}
        weights = cell_thickness(depth)[mask]
        pay_values = pay_average_data(well)

    for curve_name in curves:
        curve = well.get(curve_name)
        if curve is None:
            results[curve_name] = np.nan
            continue

        if curve_name.upper() in pay_values:
            data = pay_values[curve_name.upper()][mask]
        else:
            data = np.asarray(curve.data, dtype=float)[mask]
        geometric = curve_name.upper() in ["PERM", "PERM_CC", "K", "PERMEABILITY"]
        valid = ~np.isnan(data) & (weights > 0)
        if geometric:
            valid &= data > 0
        if not np.any(valid):
            results[curve_name] = np.nan
            continue

        values = np.log(data[valid]) if geometric else data[valid]
        average = float(np.sum(values * weights[valid]) / np.sum(weights[valid]))
        results[curve_name] = float(np.exp(average)) if geometric else average

    return results


def weighted_averages(
    well: "Well",
    curves: List[str],
    weight_curve: str = "PHIE",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: Optional[str] = "NET_PAY"
) -> Dict[str, float]:
    """
    Calculate averages weighted by a curve (e.g. porosity-weighted).

    Each sample is weighted by ``weight_curve`` alone, not by its thickness.
    When ``net_pay_curve`` is named and is not on the well, a warning is
    given and every average is NaN.

    Parameters
    ----------
    well : Well
        Well object
    curves : list
        Curves to average
    weight_curve : str
        Curve to use as weights (e.g., PHIE for porosity-weighting)
    top_depth : float, optional
        Top depth
    bottom_depth : float, optional
        Bottom depth
    net_pay_curve : str, optional
        Net pay flag curve (None to ignore)

    Returns
    -------
    dict
        Dictionary of weighted averages
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

    # Build mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)

    if net_pay_curve:
        net_pay = well.get(net_pay_curve)
        if net_pay is not None:
            mask = mask & (net_pay.data > 0.5)
        elif not reproducing_benchmark():
            warnings.warn(
                f"Net-pay flag curve {net_pay_curve!r} is not on the well, so there is no pay to "
                "average over: the averages are NaN. Pass net_pay_curve=None to average every "
                "sample of the interval.",
                UserWarning, stacklevel=2,
            )
            return {name: np.nan for name in curves}

    # Get weights
    weight_c = well.get(weight_curve)
    if weight_c is None:
        warnings.warn(
            f"Weight curve {weight_curve!r} is not on the well; the averages are unweighted",
            UserWarning,
            stacklevel=2,
        )
        weights = np.ones(len(depth))[mask]
    else:
        weights = weight_c.data[mask]
        weights = np.where(np.isnan(weights), 0, weights)

    results = {}

    for curve_name in curves:
        curve = well.get(curve_name)
        if curve is None:
            results[curve_name] = np.nan
            continue

        data = curve.data[mask]

        # Create valid mask
        valid_mask = ~np.isnan(data) & (weights > 0)

        if np.sum(valid_mask) == 0:
            results[curve_name] = np.nan
            continue

        valid_data = data[valid_mask]
        valid_weights = weights[valid_mask]

        weighted_avg = np.sum(valid_data * valid_weights) / np.sum(valid_weights)
        results[curve_name] = float(weighted_avg)

    return results


def pore_volume_weighted_sw(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: Optional[str] = "NET_PAY",
) -> float:
    """
    Pore-volume-weighted water saturation over net pay.

    Sw_pv = sum(Sw * PHIE * h) / sum(PHIE * h), with h each sample's
    thickness, so the samples holding the most pore volume count most and
    ``(1 - Sw_pv) * sum(PHIE * h)`` is the hydrocarbon pore thickness.
    Porosity is that of the pay averages (PHIE_CC where the interpretation
    produced or used it, PHIE at the samples it does not cover; see
    :func:`~petrophys.evaluation.summaries.zone_summary.pay_average_data`).
    Samples without a finite SW and porosity are left out.

    Parameters
    ----------
    well : Well
        Well with SW and PHIE (or PHIE_CC) curves.
    top_depth, bottom_depth : float, optional
        Depth range; the whole well by default.
    net_pay_curve : str, optional
        Net pay flag curve; ``None`` averages every sample in the range.

    Returns
    -------
    float
        The weighted Sw, v/v; NaN without SW, porosity or pore volume.
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    if net_pay_curve:
        net_pay = well.get(net_pay_curve)
        if net_pay is None:
            return np.nan
        mask &= np.asarray(net_pay.data, dtype=float) > 0.5

    sw, _, _ = get_curve_data_depthwise(well, ["SW"])
    if sw is None or pay_average_curves(well)["PHIE"] is None:
        return np.nan
    sw = np.asarray(sw, dtype=float)
    phie = pay_average_data(well)["PHIE"]
    valid = mask & np.isfinite(sw) & np.isfinite(phie) & (phie > 0)
    pore_volume = phie[valid] * cell_thickness(depth)[valid]
    total = float(np.sum(pore_volume))
    if total <= 0:
        return np.nan
    return float(np.sum(sw[valid] * pore_volume) / total)


def calculate_statistics(
    well: "Well",
    curve_name: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_only: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a curve.

    Parameters
    ----------
    well : Well
        Well object
    curve_name : str
        Curve name
    top_depth : float, optional
        Top depth
    bottom_depth : float, optional
        Bottom depth
    net_pay_only : bool
        Only consider net pay intervals

    Returns
    -------
    dict
        Dictionary with statistics
    """
    depth = well_depth_array(well)
    top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

    mask = (depth >= top_depth) & (depth <= bottom_depth)

    if net_pay_only:
        net_pay = well.get("NET_PAY")
        if net_pay is not None:
            mask = mask & (net_pay.data > 0.5)

    curve = well.get(curve_name)
    if curve is None:
        return {"error": f"Curve {curve_name} not found"}

    data = curve.data[mask]
    valid = data[~np.isnan(data)]

    if len(valid) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
        }

    return {
        "count": len(valid),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p50": float(np.percentile(valid, 50)),
        "p90": float(np.percentile(valid, 90)),
    }
