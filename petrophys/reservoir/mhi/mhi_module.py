"""
Moveable Hydrocarbon Index calculation module.
"""

from typing import Optional, Union
import numpy as np


class MhiModule:
    """Moveable Hydrocarbon Index calculator."""

    def calculate(
        self,
        well: "Well",
        sxo: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate MHI."""
        if sxo is None:
            sxo_curve = well.get("SXO")
            if sxo_curve is None:
                raise ValueError("SXO required")
            sxo = sxo_curve.data

        if sw is None:
            sw_curve = well.get("SW")
            if sw_curve is None:
                raise ValueError("SW required")
            sw = sw_curve.data

        return calculate_mhi(sxo, sw)

    def add_to_well(self, well: "Well") -> "Well":
        """Add MHI curve to well."""
        mhi = self.calculate(well)
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")
        return well


def calculate_mhi(
    sxo: Union[float, np.ndarray],
    sw: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Moveable Hydrocarbon Index.

    MHI = Sxo / Sw

    Parameters
    ----------
    sxo : float or np.ndarray
        Flushed zone water saturation
    sw : float or np.ndarray
        Uninvaded zone water saturation

    Returns
    -------
    float or np.ndarray
        Moveable hydrocarbon index
    """
    # Avoid division by zero
    if isinstance(sw, np.ndarray):
        sw_safe = np.where(sw <= 0, 0.01, sw)
    else:
        sw_safe = max(sw, 0.01)

    mhi = sxo / sw_safe
    return mhi
