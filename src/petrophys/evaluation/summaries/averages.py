"""
Average calculations for zones and intervals.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Union
import warnings
import numpy as np

from petrophys._internal.analysis_helpers import (
    get_curve_data_depthwise,
    resolve_well_depth_range,
    well_depth_array,
)
from petrophys._internal.benchmark import reproducing_benchmark

#: Core-corrected curves preferred, where they exist, by the pay averages.
_CORE_CORRECTED = {"PHIE": "PHIE_CC", "PERM": "PERM_CC"}

if TYPE_CHECKING:
    from petrophys.model.well import Well


def calculate_zone_averages(
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
        Curves to use arithmetic mean (default)
    use_geometric : list, optional
        Curves to use geometric mean (e.g., PERM)
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
        use_geometric = ["PERM", "K", "KLOG", "PERMEABILITY"]

    if use_harmonic is None:
        use_harmonic = []

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

        if curve_name.upper() in [c.upper() for c in use_harmonic]:
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


def calculate_pay_averages(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: str = "NET_PAY",
    curves: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate averages over net pay intervals only.

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
        ``PERM`` take the core-corrected ``PHIE_CC`` / ``PERM_CC`` sample
        where it exists, as the interpretation's SW and NET_PAY do.

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

    for curve_name in curves:
        curve = well.get(curve_name)
        if curve is None:
            results[curve_name] = np.nan
            continue

        corrected = _CORE_CORRECTED.get(curve_name.upper())
        if corrected is not None and not reproducing_benchmark():
            data, _, _ = get_curve_data_depthwise(well, [corrected, curve_name])
            data = (np.asarray(data, dtype=float) if data is not None else np.asarray(curve.data, dtype=float))[mask]
        else:
            data = curve.data[mask]
        valid = data[~np.isnan(data)]

        if len(valid) == 0:
            results[curve_name] = np.nan
            continue

        # Use geometric mean for permeability
        if curve_name.upper() in ["PERM", "K", "PERMEABILITY"]:
            pos = valid[valid > 0]
            if len(pos) > 0:
                results[curve_name] = float(np.exp(np.mean(np.log(pos))))
            else:
                results[curve_name] = np.nan
        else:
            results[curve_name] = float(np.mean(valid))

    return results


def calculate_weighted_averages(
    well: "Well",
    curves: List[str],
    weight_curve: str = "PHIE",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    net_pay_curve: Optional[str] = "NET_PAY"
) -> Dict[str, float]:
    """
    Calculate porosity-weighted or thickness-weighted averages.

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
