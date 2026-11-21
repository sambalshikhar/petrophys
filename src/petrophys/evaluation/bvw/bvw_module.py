"""
Bulk Volume Water calculation module.

References
----------
Buckles, R.S. (1965). Correlating and averaging connate water saturation
data. *Journal of Canadian Petroleum Technology* 4(1).
"""

from typing import TYPE_CHECKING, Optional, Union
import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.validation import broadcast_inputs, warn_range

if TYPE_CHECKING:
    from petrophys.model.well import Well


class BvwModule:
    """Bulk Volume Water calculator.

    Porosity is the core-corrected ``PHIE_CC`` where it exists, else ``PHIE``,
    else ``PHIT`` (the same porosity the interpretation's SW and NET_PAY use;
    the benchmark reproduction reads PHIE/PHIT).
    """

    def calculate(
        self,
        well: "Well",
        phie: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate BVW."""
        if phie is None:
            names = ["PHIE", "PHIT"] if reproducing_benchmark() else ["PHIE_CC", "PHIE", "PHIT"]
            phie, _, _ = get_curve_data_depthwise(well, names)
            if phie is None:
                raise ValueError(f"{' or '.join(names)} with finite samples required")

        if sw is None:
            sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
            if sw is None:
                raise ValueError("SW with finite samples required")

        return calculate_bvw(phie, sw)

    def add_to_well(self, well: "Well") -> "Well":
        """Add a BVW curve to the well; an existing BVW (the interpretation's) is kept."""
        if well.get("BVW") is not None and not reproducing_benchmark():
            return well
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
    phie, sw = broadcast_inputs(phie=phie, sw=sw)
    warn_range(phie, "BVW porosity (fraction)", 0, 1)
    warn_range(sw, "BVW water saturation (fraction)", 0, 1)
    result = phie * sw
    warn_range(result, "BVW output (fraction)", 0, 1)
    return float(result) if result.ndim == 0 else result
