"""
Bulk Volume Water calculation module.
"""

from typing import Optional, Union
import numpy as np

from petrophys.utils.analysis_helpers import get_curve_data_depthwise


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
            phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
            if phie is None:
                raise ValueError("PHIE or PHIT with finite samples required")

        if sw is None:
            sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
            if sw is None:
                raise ValueError("SW with finite samples required")

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
