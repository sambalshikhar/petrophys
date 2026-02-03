"""
Bulk Volume Water calculation module.
"""

from typing import Optional, Union
import numpy as np


class BvwModule:
    """Bulk Volume Water calculator."""

    def calculate(
        self,
        well: "Well",
        phie: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate BVW."""
        if phie is None:
            phie_curve = well.get("PHIE")
            if phie_curve is None:
                raise ValueError("PHIE required")
            phie = phie_curve.data

        if sw is None:
            sw_curve = well.get("SW")
            if sw_curve is None:
                raise ValueError("SW required")
            sw = sw_curve.data

        return phie * sw

    def add_to_well(self, well: "Well") -> "Well":
        """Add BVW curve to well."""
        bvw = self.calculate(well)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")
        return well


def calculate_bvw(
    phie: Union[float, np.ndarray],
    sw: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Bulk Volume Water.

    BVW = φe × Sw

    Parameters
    ----------
    phie : float or np.ndarray
        Effective porosity
    sw : float or np.ndarray
        Water saturation

    Returns
    -------
    float or np.ndarray
        Bulk volume water
    """
    return phie * sw
