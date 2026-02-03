"""
Validation utilities for petrophysics data.
"""

from typing import Optional, Tuple, List, Union, Any
import numpy as np


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_array(
    data: Any,
    name: str = "data",
    min_length: int = 1,
    allow_nan: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> np.ndarray:
    """
    Validate and convert data to numpy array.

    Parameters
    ----------
    data : Any
        Input data
    name : str
        Name for error messages
    min_length : int
        Minimum array length
    allow_nan : bool
        Whether to allow NaN values
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value

    Returns
    -------
    np.ndarray
        Validated array

    Raises
    ------
    ValidationError
        If validation fails
    """
    if data is None:
        raise ValidationError(f"{name} cannot be None")

    try:
        arr = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} cannot be converted to array: {e}")

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if len(arr) < min_length:
        raise ValidationError(
            f"{name} must have at least {min_length} elements, got {len(arr)}"
        )

    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError(f"{name} contains NaN values")

    valid_mask = ~np.isnan(arr)

    if min_value is not None and np.any(arr[valid_mask] < min_value):
        raise ValidationError(f"{name} contains values below {min_value}")

    if max_value is not None and np.any(arr[valid_mask] > max_value):
        raise ValidationError(f"{name} contains values above {max_value}")

    return arr


def validate_range(
    value: Union[float, int],
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False
) -> Optional[float]:
    """
    Validate a scalar value is within range.

    Parameters
    ----------
    value : float or int
        Value to validate
    name : str
        Name for error messages
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
    allow_none : bool
        Whether to allow None values

    Returns
    -------
    float or None
        Validated value

    Raises
    ------
    ValidationError
        If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"{name} cannot be None")

    try:
        value = float(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} must be numeric: {e}")

    if np.isnan(value):
        raise ValidationError(f"{name} cannot be NaN")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")

    return value


def validate_curve(
    well: "Well",
    curve_name: str,
    required: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    Validate that a curve exists in a well and get its data.

    Parameters
    ----------
    well : Well
        Well object
    curve_name : str
        Name of the curve
    required : bool
        Whether the curve is required
    min_value : float, optional
        Minimum valid value
    max_value : float, optional
        Maximum valid value

    Returns
    -------
    np.ndarray or None
        Curve data if found

    Raises
    ------
    ValidationError
        If validation fails
    """
    curve = well.get(curve_name)

    if curve is None:
        if required:
            raise ValidationError(f"Required curve '{curve_name}' not found in well")
        return None

    data = curve.data

    if min_value is not None or max_value is not None:
        valid_mask = ~np.isnan(data)
        if min_value is not None and np.any(data[valid_mask] < min_value):
            raise ValidationError(
                f"Curve '{curve_name}' contains values below {min_value}"
            )
        if max_value is not None and np.any(data[valid_mask] > max_value):
            raise ValidationError(
                f"Curve '{curve_name}' contains values above {max_value}"
            )

    return data


def validate_well(
    well: "Well",
    required_curves: Optional[List[str]] = None,
    min_samples: int = 1
) -> bool:
    """
    Validate a well object.

    Parameters
    ----------
    well : Well
        Well object to validate
    required_curves : list, optional
        List of required curve names
    min_samples : int
        Minimum number of samples

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    if well is None:
        raise ValidationError("Well cannot be None")

    if not hasattr(well, 'n_samples'):
        raise ValidationError("Invalid well object: missing n_samples")

    if well.n_samples < min_samples:
        raise ValidationError(
            f"Well must have at least {min_samples} samples, got {well.n_samples}"
        )

    if required_curves:
        missing = []
        for curve_name in required_curves:
            if well.get(curve_name) is None:
                missing.append(curve_name)
        if missing:
            raise ValidationError(
                f"Missing required curves: {', '.join(missing)}"
            )

    return True


def validate_depth_range(
    top: float,
    bottom: float,
    well: Optional["Well"] = None
) -> Tuple[float, float]:
    """
    Validate a depth range.

    Parameters
    ----------
    top : float
        Top depth
    bottom : float
        Bottom depth
    well : Well, optional
        Well to check against

    Returns
    -------
    tuple
        (top, bottom) validated

    Raises
    ------
    ValidationError
        If validation fails
    """
    top = validate_range(top, "top_depth", min_value=0)
    bottom = validate_range(bottom, "bottom_depth", min_value=0)

    if top >= bottom:
        raise ValidationError(f"Top depth ({top}) must be less than bottom ({bottom})")

    if well is not None:
        well_top = well.start_depth
        well_bottom = well.stop_depth

        if well_top is not None and top < well_top:
            raise ValidationError(
                f"Top depth ({top}) is above well start ({well_top})"
            )

        if well_bottom is not None and bottom > well_bottom:
            raise ValidationError(
                f"Bottom depth ({bottom}) is below well stop ({well_bottom})"
            )

    return top, bottom


def validate_cutoffs(
    vcl_cutoff: Optional[float] = None,
    phie_cutoff: Optional[float] = None,
    sw_cutoff: Optional[float] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Validate net pay cutoff values.

    Parameters
    ----------
    vcl_cutoff : float, optional
        VCL cutoff (0-1)
    phie_cutoff : float, optional
        PHIE cutoff (0-1)
    sw_cutoff : float, optional
        Sw cutoff (0-1)

    Returns
    -------
    tuple
        Validated (vcl_cutoff, phie_cutoff, sw_cutoff)
    """
    if vcl_cutoff is not None:
        vcl_cutoff = validate_range(vcl_cutoff, "vcl_cutoff", min_value=0, max_value=1)

    if phie_cutoff is not None:
        phie_cutoff = validate_range(phie_cutoff, "phie_cutoff", min_value=0, max_value=1)

    if sw_cutoff is not None:
        sw_cutoff = validate_range(sw_cutoff, "sw_cutoff", min_value=0, max_value=1)

    return vcl_cutoff, phie_cutoff, sw_cutoff
