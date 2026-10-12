"""
Neutron porosity correction for hydrocarbon effect.

Design doc equations:
    Step 1 — Preliminary neutron porosity:
        phi_N_prelim = (NPHI - VCL * PHI_CL) / (Sxo + (1 - Sxo) * NeuHyHI)

    Step 2 — Corrected NPHI:
        NPHI_HC = NPHI + phi_N_prelim * (1 - Sxo) * (1 - NeuHyHI)

NeuHyHI is the hydrocarbon hydrogen index: DefaultParameters.OIL_HYDROGEN_INDEX
(1.0, so oil leaves the neutron unchanged) or GAS_HYDROGEN_INDEX.
"""

from typing import Union, Optional
import numpy as np

from petrophys.model.constants import DefaultParameters


def correct_neutron_hc(
    nphi: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    hc_type: str = "oil",
    vcl: Optional[Union[float, np.ndarray]] = None,
    phi_cl: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Correct neutron porosity for hydrocarbon effect.

    When VCL is provided, uses the design doc's preliminary porosity
    approach. Otherwise falls back to the simpler direct correction.

    Parameters
    ----------
    nphi : float or np.ndarray
        Raw neutron porosity (v/v)
    phi : float or np.ndarray
        Porosity estimate (used in fallback mode)
    sxo : float or np.ndarray
        Flushed zone water saturation (v/v)
    hc_type : str
        Hydrocarbon type ("oil" or "gas")
    vcl : float or np.ndarray, optional
        Volume of clay (fraction). If provided, uses preliminary
        porosity approach from design doc.
    phi_cl : float
        Clay porosity from NPHI at 100% clay (default: 0.15)

    Returns
    -------
    float or np.ndarray
        HC-corrected neutron porosity
    """
    neu_hy_hi = (
        DefaultParameters.GAS_HYDROGEN_INDEX
        if hc_type.lower() == "gas"
        else DefaultParameters.OIL_HYDROGEN_INDEX
    )

    if vcl is not None:
        # Design doc approach with preliminary porosity
        denominator = sxo + (1.0 - sxo) * neu_hy_hi
        if isinstance(denominator, np.ndarray):
            denominator = np.where(denominator <= 0, 1e-10, denominator)
        else:
            denominator = max(denominator, 1e-10)

        phi_n_prelim = (nphi - vcl * phi_cl) / denominator
        nphi_hc = nphi + phi_n_prelim * (1.0 - sxo) * (1.0 - neu_hy_hi)
    else:
        # Simpler fallback
        nphi_hc = nphi + phi * (1.0 - sxo) * (1.0 - neu_hy_hi)

    return nphi_hc
