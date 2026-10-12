"""
Flag generation for bad hole conditions.

This module provides functions to generate individual quality flags
for various bad hole conditions.
"""

from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


def _plausible_bit_size(value: Any) -> Optional[float]:
    """A header bit size in inches, or None when it cannot be one.

    Bits run from about 3 in (slim hole) to 36 in (conductor hole); values
    outside 3-36 in are a null such as -999.25 or another unit, such as a
    12.25 in bit written as 311.15 mm.
    """
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    return size if 3.0 <= size <= 36.0 else None


def _known_bit_size(well: "Well") -> np.ndarray:
    """The bit size of every sample in inches, NaN where the well records none.

    The BS curve where it has a value, else the header bit size when it is a
    hole size in inches (:func:`_plausible_bit_size`). No size is assumed.
    """
    bit_size, _, _ = get_curve_data_depthwise(well, ["BS"])
    if bit_size is None:
        bit_size = np.full(well.n_samples, np.nan)
    header = _plausible_bit_size(getattr(well.header, "bit_size", None))
    if header is not None:
        bit_size = np.where(np.isfinite(bit_size), bit_size, header)
    return np.asarray(bit_size, dtype=float)


def generate_washout_flag(
    cali: np.ndarray,
    bit_size: Union[float, np.ndarray],
    drho: Optional[np.ndarray] = None,
    cali_threshold: float = DefaultParameters.CALI_BS_THRESHOLD,
    drho_threshold: float = DefaultParameters.DRHO_THRESHOLD,
) -> np.ndarray:
    """
    Generate washout flag based on caliper and density correction.

    Washout is detected when:

    - CALI - BS > threshold (default 1.5 inches), OR
    - abs(DRHO) > threshold (default 0.15 g/cm3)

    Parameters
    ----------
    cali : np.ndarray
        Caliper log values (inches)
    bit_size : float or np.ndarray
        Bit size in inches. A per-depth BS curve may be supplied.
    drho : np.ndarray, optional
        Density correction log
    cali_threshold : float, optional
        Caliper-BS threshold (default: 1.5 inches)
    drho_threshold : float, optional
        DRHO threshold (default: 0.15 g/cm³)

    Returns
    -------
    np.ndarray
        Boolean flag array (True = washout)
    """
    # Caliper criterion
    cali_flag = (cali - bit_size) > cali_threshold

    # DRHO criterion (if available)
    if drho is not None:
        drho_flag = np.abs(drho) > drho_threshold
        return cali_flag | drho_flag

    return cali_flag


def generate_large_hole_flag(
    bit_size: Union[float, np.ndarray],
    threshold: float = DefaultParameters.LARGE_HOLE_BS,
) -> Union[bool, np.ndarray]:
    """
    Generate large hole flag.

    Large hole is detected when bit size > threshold (default 17.5 inches).

    Parameters
    ----------
    bit_size : float or np.ndarray
        Bit size value(s) in inches
    threshold : float, optional
        Large hole threshold (default: 17.5 inches)

    Returns
    -------
    bool or np.ndarray
        Flag indicating large hole condition
    """
    return bit_size > threshold


def generate_low_density_flag(
    rhob: np.ndarray, threshold: float = DefaultParameters.RHOB_MIN
) -> np.ndarray:
    """
    Generate low density flag.

    Low density is detected when RHOB < threshold (default 2.1 g/cm³).

    Parameters
    ----------
    rhob : np.ndarray
        Bulk density log (g/cm³)
    threshold : float, optional
        Low density threshold (default: 2.1 g/cm³)

    Returns
    -------
    np.ndarray
        Boolean flag array (True = low density)
    """
    return rhob < threshold


def generate_pef_exclusion_flag(
    pef: np.ndarray,
    max_threshold: float = DefaultParameters.PEF_MAX,
    min_threshold: float = DefaultParameters.PEF_MIN,
) -> np.ndarray:
    """
    Generate PEF exclusion flag.

    PEF is excluded when:
    - PEF > max_threshold (default 7), OR
    - PEF < min_threshold (default 0.5)

    Parameters
    ----------
    pef : np.ndarray
        Photoelectric factor log
    max_threshold : float, optional
        Maximum PEF threshold (default: 7)
    min_threshold : float, optional
        Minimum PEF threshold (default: 0.5)

    Returns
    -------
    np.ndarray
        Boolean flag array (True = exclude PEF)
    """
    return (pef > max_threshold) | (pef < min_threshold)


def generate_bad_hole_flags(
    well: "Well",
    bit_size: Optional[Union[float, np.ndarray]] = None,
    cali_threshold: float = DefaultParameters.CALI_BS_THRESHOLD,
    drho_threshold: float = DefaultParameters.DRHO_THRESHOLD,
    rhob_min: float = DefaultParameters.RHOB_MIN,
    pef_max: float = DefaultParameters.PEF_MAX,
    pef_min: float = DefaultParameters.PEF_MIN,
    large_hole_threshold: float = DefaultParameters.LARGE_HOLE_BS,
    *,
    drho_alone: bool = False,
) -> dict:
    """
    Generate all bad hole flags for a well.

    Parameters
    ----------
    well : Well
        Well object with log curves
    bit_size : float or np.ndarray, optional
        Bit size in inches. If not supplied, uses the header value or the
        depth-by-depth BS curve when available.
    cali_threshold : float, optional
        Caliper-BS washout threshold
    drho_threshold : float, optional
        DRHO washout threshold
    rhob_min : float, optional
        Low density threshold
    pef_max : float, optional
        Maximum valid PEF
    pef_min : float, optional
        Minimum valid PEF
    large_hole_threshold : float, optional
        Large hole BS threshold
    drho_alone : bool, default False
        Also flag washout from ``|DRHO|`` alone when the caliper test cannot
        run: the well has no caliper curve, or no bit size. By default
        ``|DRHO|`` is tested only when the well has both.

    Returns
    -------
    dict
        Dictionary of flag arrays:
        - 'washout': Washout flag
        - 'large_hole': Large hole flag
        - 'low_density': Low density screening flag
        - 'pef_exclusion': PEF exclusion flag
        - 'bad_hole': Combined bad hole flag
    """
    n_samples = well.n_samples
    flags = {
        "washout": np.zeros(n_samples, dtype=bool),
        "large_hole": np.zeros(n_samples, dtype=bool),
        "low_density": np.zeros(n_samples, dtype=bool),
        "pef_exclusion": np.zeros(n_samples, dtype=bool),
    }

    # Get bit size
    if bit_size is None:
        bit_size = well.header.bit_size
    if bit_size is None:
        # Try to get from BS curve
        bs_data, _, _ = get_curve_data_depthwise(well, ["BS"])
        if bs_data is not None:
            bit_size = bs_data

    if bit_size is not None and not np.isscalar(bit_size):
        bit_size = np.asarray(bit_size, dtype=float)
        if bit_size.shape != (n_samples,):
            finite = bit_size[np.isfinite(bit_size)]
            bit_size = float(np.nanmedian(finite)) if finite.size else None

    # Generate washout flag: (CALI - BS) or |DRHO| over threshold, evaluated
    # where both a caliper and a bit size are available; with drho_alone,
    # |DRHO| over threshold also without them.
    cali, _, _ = get_curve_data_depthwise(well, ["CALI", "CAL"])
    drho, _, _ = get_curve_data_depthwise(well, ["DRHO"])

    if cali is not None and bit_size is not None:
        flags["washout"] = generate_washout_flag(cali, bit_size, drho, cali_threshold, drho_threshold)
    elif drho_alone and drho is not None:
        flags["washout"] = np.abs(drho) > drho_threshold

    # Generate large hole flag
    if bit_size is not None:
        large_hole = generate_large_hole_flag(bit_size, large_hole_threshold)
        if np.isscalar(large_hole):
            flags["large_hole"] = np.full(n_samples, bool(large_hole))
        else:
            flags["large_hole"] = np.asarray(large_hole, dtype=bool)

    # Generate low density flag
    rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
    if rhob is not None:
        flags["low_density"] = generate_low_density_flag(rhob, rhob_min)

    # Generate PEF exclusion flag
    pef, _, _ = get_curve_data_depthwise(well, ["PEF"])
    if pef is not None:
        flags["pef_exclusion"] = generate_pef_exclusion_flag(pef, pef_max, pef_min)

    # Combined bad hole flag
    #
    # BAD_HOLE intentionally excludes the low-density screen so the
    # library's bad-hole output is driven only by caliper/DRHO washout
    # and large-hole geometry criteria. The low-density flag is still
    # returned separately for QC reference.
    flags["bad_hole"] = flags["washout"] | flags["large_hole"]

    return flags
