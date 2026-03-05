"""
Clay Resistivity (Rcl) calculation module.

Computes clay resistivity from deep resistivity readings in
zones where clay volume exceeds 0.9 (nearly pure clay).
"""

from typing import Union, Optional
import numpy as np


def calculate_rcl(
    rdep: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    vcl_threshold: float = 0.9,
    default_rcl: float = 5.0
) -> float:
    """
    Calculate clay resistivity from high-VCL zones.

    Filters for zones where VCL > threshold (nearly pure clay),
    then averages the deep resistivity readings.

    Parameters
    ----------
    rdep : float or np.ndarray
        Deep resistivity log (ohm.m)
    vcl : float or np.ndarray
        Final VCLAY curve (fraction)
    vcl_threshold : float
        VCL threshold for pure clay identification (default: 0.9)
    default_rcl : float
        Default Rcl if no pure clay zones found (default: 5.0 ohm.m)

    Returns
    -------
    float
        Average Rcl (ohm.m)
    """
    if isinstance(vcl, np.ndarray) and isinstance(rdep, np.ndarray):
        pure_clay_mask = vcl > vcl_threshold
        valid_mask = pure_clay_mask & ~np.isnan(rdep) & ~np.isnan(vcl)

        if np.any(valid_mask):
            return float(np.nanmean(rdep[valid_mask]))

    return default_rcl


class RclModule:
    """Clay resistivity calculator."""

    def __init__(self, vcl_threshold: float = 0.9, default_rcl: float = 5.0):
        self.vcl_threshold = vcl_threshold
        self.default_rcl = default_rcl

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Rcl from well data.

        Parameters
        ----------
        well : Well
            Well object with RDEP and VCL curves
        vcl : np.ndarray, optional
            Clay volume array (uses well VCL if not provided)

        Returns
        -------
        float
            Clay resistivity (ohm.m)
        """
        rdep = well.get("RDEP") or well.get("RT")
        if rdep is None:
            return self.default_rcl

        if vcl is None:
            vcl_curve = well.get("VCL")
            if vcl_curve is None:
                return self.default_rcl
            vcl = vcl_curve.data

        return calculate_rcl(rdep.data, vcl, self.vcl_threshold, self.default_rcl)
