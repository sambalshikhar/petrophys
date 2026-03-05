"""
Density correction for hydrocarbon effect.

Design doc equations:
    Step 1 — Preliminary density porosity:
        phi_D_prelim = (rho_ma - RHOB - VCL * (rho_ma - RHOB)) /
                       (rho_ma - RHOB * Sxo - Rho_HC * (1 - Sxo))

    Step 2 — Corrected RHOB:
        RHOB_HC = RHOB + phi_D_prelim * (1 - Sxo) * (Rho_fl - Rho_HC)
"""

from typing import Union, Optional
import numpy as np


def correct_density_hc(
    rhob: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    rhob_fluid: float = 1.0,
    hc_type: str = "oil",
    vcl: Optional[Union[float, np.ndarray]] = None,
    rho_ma: float = 2.65
) -> Union[float, np.ndarray]:
    """
    Correct bulk density for hydrocarbon effect.

    When VCL and rho_ma are provided, uses the design doc's preliminary
    porosity approach. Otherwise falls back to the simpler direct correction.

    Parameters
    ----------
    rhob : float or np.ndarray
        Raw bulk density (g/cc)
    phi : float or np.ndarray
        Porosity estimate (used in fallback mode)
    sxo : float or np.ndarray
        Flushed zone water saturation (v/v)
    rhob_fluid : float
        Fluid density in flushed zone (g/cc)
    hc_type : str
        Hydrocarbon type ("oil" or "gas")
    vcl : float or np.ndarray, optional
        Volume of clay (fraction). If provided, uses preliminary
        porosity approach from design doc.
    rho_ma : float
        Matrix density (g/cc, default: 2.65)

    Returns
    -------
    float or np.ndarray
        HC-corrected bulk density
    """
    rho_hc = 0.2 if hc_type.lower() == "gas" else 0.8

    if vcl is not None:
        # Design doc approach with preliminary density porosity
        numerator = rho_ma - rhob - vcl * (rho_ma - rhob)
        denominator = rho_ma - rhob * sxo - rho_hc * (1.0 - sxo)

        if isinstance(denominator, np.ndarray):
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        else:
            denominator = denominator if abs(denominator) > 1e-10 else 1e-10

        phi_d_prelim = numerator / denominator
        rhob_corrected = rhob + phi_d_prelim * (1.0 - sxo) * (rhob_fluid - rho_hc)
    else:
        # Simpler fallback
        rhob_corrected = rhob + phi * (1.0 - sxo) * (rhob_fluid - rho_hc)

    return rhob_corrected
