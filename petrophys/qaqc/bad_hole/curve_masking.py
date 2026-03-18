"""
Curve masking for bad hole zones.

This module provides functions to mask or remove unreliable
curve data in bad hole intervals.
"""

from typing import Optional, List
import numpy as np


# Curves affected by bad hole conditions (shallow measurements)
SHALLOW_CURVES = ["RHOB", "PEF", "NPHI", "DRHO", "CALI"]

# Deep measurements (less affected by bad hole)
DEEP_CURVES = ["RDEP", "RMED", "GR", "DTC", "DTS"]


def mask_bad_hole_curves(
    well: "Well",
    bad_hole_flag: np.ndarray,
    curves_to_mask: Optional[List[str]] = None,
    mask_value: float = np.nan,
    inplace: bool = False
) -> "Well":
    """
    Mask curves in bad hole zones.

    For bad hole/washout intervals, shallow curves (RHOB, PEF, NPHI)
    are typically unreliable and should be masked.

    Parameters
    ----------
    well : Well
        Well object
    bad_hole_flag : np.ndarray
        Boolean array indicating bad hole zones
    curves_to_mask : list of str, optional
        Curves to mask (default: SHALLOW_CURVES)
    mask_value : float, optional
        Value to use for masked data (default: NaN)
    inplace : bool, optional
        Modify well in place (default: False)

    Returns
    -------
    Well
        Well with masked curves
    """
    if not inplace:
        # Create shallow copy of well
        from petrophys.base.well import Well
        new_well = Well(name=well.name)
        new_well._depth = well._depth.copy() if well._depth is not None else None
        new_well._depth_unit = well._depth_unit
        new_well.header = well.header
        new_well._parameters = well._parameters.copy()
        new_well._results = well._results.copy()
        for mnemonic, curve in well._curves.items():
            new_well._curves[mnemonic] = curve.copy()
        well = new_well

    # Determine curves to mask
    if curves_to_mask is None:
        curves_to_mask = SHALLOW_CURVES

    # Apply mask
    for mnemonic in curves_to_mask:
        mnemonic = mnemonic.upper()
        if mnemonic in well._curves:
            curve = well._curves[mnemonic]
            curve._data = np.where(bad_hole_flag, mask_value, curve._data)

    return well


def apply_quality_flags(
    well: "Well",
    flags: dict,
    mask_shallow: bool = True,
    mask_pef: bool = True,
    inplace: bool = False
) -> "Well":
    """
    Apply quality flags to mask unreliable data.

    Parameters
    ----------
    well : Well
        Well object
    flags : dict
        Dictionary of flag arrays from generate_bad_hole_flags
    mask_shallow : bool, optional
        Mask shallow curves in washout zones (default: True)
    mask_pef : bool, optional
        Mask PEF where exclusion flag is set (default: True)
    inplace : bool, optional
        Modify well in place (default: False)

    Returns
    -------
    Well
        Well with quality flags applied
    """
    if not inplace:
        # Create copy
        from petrophys.base.well import Well
        new_well = Well(name=well.name)
        new_well._depth = well._depth.copy() if well._depth is not None else None
        new_well._depth_unit = well._depth_unit
        new_well.header = well.header
        new_well._parameters = well._parameters.copy()
        new_well._results = well._results.copy()
        for mnemonic, curve in well._curves.items():
            new_well._curves[mnemonic] = curve.copy()
        well = new_well

    # Mask shallow curves in bad hole zones
    if mask_shallow and "bad_hole" in flags:
        bad_hole_flag = flags["bad_hole"]
        for mnemonic in SHALLOW_CURVES:
            if mnemonic in well._curves:
                curve = well._curves[mnemonic]
                curve._data = np.where(bad_hole_flag, np.nan, curve._data)

    # Mask PEF separately (can have different criteria)
    if mask_pef and "pef_exclusion" in flags:
        pef_flag = flags["pef_exclusion"]
        if "PEF" in well._curves:
            curve = well._curves["PEF"]
            curve._data = np.where(pef_flag, np.nan, curve._data)

    return well


def create_quality_curve(
    well: "Well",
    flags: dict,
    weights: Optional[dict] = None
) -> np.ndarray:
    """
    Create a composite quality curve from flags.

    Parameters
    ----------
    well : Well
        Well object
    flags : dict
        Dictionary of flag arrays
    weights : dict, optional
        Weights for each flag type (default: equal weights)

    Returns
    -------
    np.ndarray
        Quality curve (0 = best, higher = worse)
    """
    n_samples = well.n_samples
    quality = np.zeros(n_samples, dtype=float)

    if weights is None:
        weights = {
            "washout": 1.0,
            "large_hole": 0.5,
            "low_density": 1.0,
            "pef_exclusion": 0.3,
        }

    for name, weight in weights.items():
        if name in flags:
            quality += flags[name].astype(float) * weight

    # Normalize to 0-1 range
    max_quality = sum(weights.values())
    if max_quality > 0:
        quality = quality / max_quality

    return quality


def get_good_data_mask(
    well: "Well",
    flags: dict,
    required_curves: Optional[List[str]] = None
) -> np.ndarray:
    """
    Get mask indicating good quality data points.

    Parameters
    ----------
    well : Well
        Well object
    flags : dict
        Dictionary of flag arrays
    required_curves : list of str, optional
        Curves that must have valid data

    Returns
    -------
    np.ndarray
        Boolean mask (True = good data)
    """
    n_samples = well.n_samples

    # Start with all good
    good_mask = np.ones(n_samples, dtype=bool)

    # Exclude bad hole zones
    if "bad_hole" in flags:
        good_mask &= ~flags["bad_hole"]

    # Check required curves have valid data
    if required_curves:
        for mnemonic in required_curves:
            if mnemonic in well._curves:
                curve_data = well._curves[mnemonic].data
                good_mask &= ~np.isnan(curve_data)

    return good_mask
