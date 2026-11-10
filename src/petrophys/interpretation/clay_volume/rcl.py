"""
Clay Resistivity (Rcl) calculation module.

Computes clay resistivity from deep resistivity readings in shale
samples, or in zones where clay volume exceeds 0.9 (nearly pure clay).
"""

from typing import TYPE_CHECKING, Union, Optional
import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


def calculate_rcl(
    rdep: Union[float, np.ndarray],
    vcl: Optional[Union[float, np.ndarray]] = None,
    shale_mask: Optional[np.ndarray] = None,
    vcl_threshold: float = 0.9,
    default_rcl: float = 5.0,
    statistic: str = "median",
) -> float:
    """
    Calculate clay resistivity from high-VCL zones.

    Uses shale samples when an explicit shale mask is available. Otherwise it
    falls back to zones where VCL > threshold (nearly pure clay). With neither
    a mask nor VCL, every finite RDEP sample is used.

    Parameters
    ----------
    rdep : float or np.ndarray
        Deep resistivity log (ohm.m)
    vcl : float or np.ndarray, optional
        Final VCLAY curve (fraction)
    shale_mask : np.ndarray, optional
        Boolean mask identifying shale samples
    vcl_threshold : float
        VCL threshold for pure clay identification (default: 0.9)
    default_rcl : float
        Default Rcl if no pure clay zones found (default: 5.0 ohm.m)
    statistic : str
        Reduction statistic to use over shale samples: "median" or "mean"

    Returns
    -------
    float
        Representative Rcl (ohm.m)
    """
    if statistic.lower() not in ("median", "mean"):
        raise ValueError(f"statistic must be 'median' or 'mean', not {statistic!r}")
    # Scalars and lists are documented inputs as well as arrays.
    rdep = np.atleast_1d(np.asarray(rdep, dtype=float))
    valid_mask = np.isfinite(rdep)
    if shale_mask is not None:
        valid_mask = valid_mask & np.atleast_1d(np.asarray(shale_mask)).astype(bool)
    elif vcl is not None:
        vcl = np.atleast_1d(np.asarray(vcl, dtype=float))
        valid_mask = valid_mask & np.isfinite(vcl) & (vcl > vcl_threshold)

    if np.any(valid_mask):
        values = rdep[valid_mask]
        if statistic.lower() == "mean":
            return float(np.nanmean(values))
        return float(np.nanmedian(values))

    return default_rcl


class RclModule:
    """Clay resistivity calculator."""

    def __init__(
        self,
        vcl_threshold: float = 0.9,
        default_rcl: float = 5.0,
        statistic: str = "median",
    ):
        self.vcl_threshold = vcl_threshold
        self.default_rcl = default_rcl
        self.statistic = statistic

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None,
        shale_mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Rcl from well data.

        Parameters
        ----------
        well : Well
            Well object with RDEP and VCL curves
        vcl : np.ndarray, optional
            Clay volume array (uses well VCL if not provided; with no VCL
            curve either, the statistic of the whole RDEP log)
        shale_mask : np.ndarray, optional
            Boolean shale-sample mask

        Returns
        -------
        float
            Clay resistivity (ohm.m)
        """
        rdep, _, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
        if rdep is None:
            return self.default_rcl

        if shale_mask is None and vcl is None:
            vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
            if vcl is None:
                return calculate_rcl(
                    rdep,
                    None,
                    shale_mask=None,
                    vcl_threshold=self.vcl_threshold,
                    default_rcl=self.default_rcl,
                    statistic=self.statistic,
                )

        return calculate_rcl(
            rdep,
            vcl,
            shale_mask=shale_mask,
            vcl_threshold=self.vcl_threshold,
            default_rcl=self.default_rcl,
            statistic=self.statistic,
        )
