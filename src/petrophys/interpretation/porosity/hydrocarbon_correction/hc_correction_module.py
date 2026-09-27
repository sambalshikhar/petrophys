"""
Main HC correction module.

Applies hydrocarbon corrections to NPHI and RHOB logs using
the design doc's preliminary porosity approach when VCL is available.
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
import numpy as np

from petrophys.interpretation.porosity.hydrocarbon_correction.neutron_correction import correct_neutron_hc
from petrophys.interpretation.porosity.hydrocarbon_correction.density_correction import correct_density_hc
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class HcCorrectionResult:
    """Result container for HC correction."""
    nphi_hc: np.ndarray
    rhob_hc: np.ndarray


class HcCorrectionModule:
    """Hydrocarbon correction for neutron and density."""

    def __init__(
        self,
        hc_type: str = "oil",
        rho_ma: float = 2.65,
        rho_fl: float = 1.0,
        phi_cl: float = 0.15,
        rho_cl: float = 2.45
    ):
        self.hc_type = hc_type
        self.rho_ma = rho_ma
        self.rho_fl = rho_fl
        self.phi_cl = phi_cl
        self.rho_cl = rho_cl

    def calculate(
        self,
        well: "Well",
        sxo: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        vcl: Optional[np.ndarray] = None
    ) -> HcCorrectionResult:
        """
        Apply HC corrections to well data.

        When VCL is available, uses the design doc's preliminary porosity
        approach for more accurate corrections.

        Parameters
        ----------
        well : Well
            Well object with NPHI and RHOB curves
        sxo : np.ndarray, optional
            Flushed zone water saturation
        phi : np.ndarray, optional
            Porosity estimate (fallback if VCL not available)
        vcl : np.ndarray, optional
            Volume of clay (enables preliminary porosity approach)
        """
        nphi, _, _ = get_curve_data_depthwise(well, ["NPHI"])
        rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])

        if nphi is None or rhob is None:
            raise ValueError("NPHI and RHOB with finite samples required")

        if sxo is None:
            sxo_data, _, _ = get_curve_data_depthwise(well, ["SXO", "SWXOA", "SXO-LOG"])
            sxo = sxo_data if sxo_data is not None else np.ones(well.n_samples)

        if vcl is None:
            vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])

        if phi is None and vcl is None:
            phi_data, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
            phi = phi_data if phi_data is not None else 0.15

        nphi_hc = correct_neutron_hc(
            nphi, phi, sxo, self.hc_type,
            vcl=vcl, phi_cl=self.phi_cl
        )
        rhob_hc = correct_density_hc(
            rhob, phi, sxo, self.rho_fl, self.hc_type,
            vcl=vcl, rho_ma=self.rho_ma, rho_cl=self.rho_cl
        )

        return HcCorrectionResult(nphi_hc=nphi_hc, rhob_hc=rhob_hc)

    def add_to_well(
        self,
        well: "Well",
        result: Optional[HcCorrectionResult] = None
    ) -> "Well":
        """Add HC-corrected curves to well."""
        if result is None:
            result = self.calculate(well)

        well.add_curve("NPHI_HC", result.nphi_hc, "v/v", "HC-Corrected Neutron Porosity")
        well.add_curve("RHOB_HC", result.rhob_hc, "g/cc", "HC-Corrected Bulk Density")

        return well


def apply_hc_correction(well: "Well", hc_type: str = "oil", **kwargs) -> HcCorrectionResult:
    """Apply HC correction."""
    module = HcCorrectionModule(hc_type=hc_type)
    return module.calculate(well, **kwargs)
