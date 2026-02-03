"""
Main HC correction module.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

from petrophys.petrophysics.hc_correction.neutron_correction import correct_neutron_hc
from petrophys.petrophysics.hc_correction.density_correction import correct_density_hc


@dataclass
class HcCorrectionResult:
    """Result container for HC correction."""
    nphi_hc: np.ndarray
    rhob_hc: np.ndarray


class HcCorrectionModule:
    """Hydrocarbon correction for neutron and density."""

    def __init__(self, hc_type: str = "oil"):
        self.hc_type = hc_type

    def calculate(
        self,
        well: "Well",
        sxo: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None
    ) -> HcCorrectionResult:
        """Apply HC corrections to well data."""
        nphi = well.get("NPHI")
        rhob = well.get("RHOB")

        if nphi is None or rhob is None:
            raise ValueError("NPHI and RHOB required")

        if sxo is None:
            sxo_curve = well.get("SXO")
            sxo = sxo_curve.data if sxo_curve is not None else np.ones(well.n_samples)

        if phi is None:
            phi_curve = well.get("PHIE") or well.get("PHIT")
            phi = phi_curve.data if phi_curve is not None else 0.15

        nphi_hc = correct_neutron_hc(nphi.data, phi, sxo, self.hc_type)
        rhob_hc = correct_density_hc(rhob.data, phi, sxo, 1.0, self.hc_type)

        return HcCorrectionResult(nphi_hc=nphi_hc, rhob_hc=rhob_hc)


def apply_hc_correction(well: "Well", hc_type: str = "oil", **kwargs) -> HcCorrectionResult:
    """Apply HC correction."""
    module = HcCorrectionModule(hc_type=hc_type)
    return module.calculate(well, **kwargs)
