"""
Density correction for hydrocarbon effect.

Step 1 - Preliminary density porosity:
    phi_D_prelim = (rho_ma - RHOB - VCL * (rho_ma - rho_cl)) /
                   (rho_ma - rho_fl * Sxo - rho_hc * (1 - Sxo))

Step 2 - Corrected RHOB:
    RHOB_HC = RHOB + phi_D_prelim * (1 - Sxo) * (rho_fl - rho_hc)

References
----------
Gaymard, R. and Poupon, A. (1968). Response of neutron and formation
density logs in hydrocarbon bearing formations. *The Log Analyst* 9(5).
"""

from typing import Optional, Union

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


def correct_density_hc(
    rhob: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    rhob_fluid: float = 1.0,
    hc_type: str = "oil",
    vcl: Optional[Union[float, np.ndarray]] = None,
    rho_ma: float = 2.65,
    rho_cl: float = 2.45,
    rho_hc: Optional[float] = None,
    floor_porosity: Optional[bool] = None,
) -> Union[float, np.ndarray]:
    """
    Correct bulk density for hydrocarbon effect.

    When ``VCL`` and ``rho_ma`` are provided, use a volumetric mixing-law
    inversion for the preliminary porosity estimate. Otherwise, fall back to
    the simpler porosity-based correction.

    Parameters
    ----------
    rhob : float or np.ndarray
        Raw bulk density (g/cc).
    phi : float or np.ndarray
        Porosity estimate used only in the simple fallback when ``VCL`` is not
        provided.
    sxo : float or np.ndarray
        Flushed-zone water saturation (v/v).
    rhob_fluid : float
        Mud-filtrate / fluid density in the flushed zone (g/cc).
    hc_type : str
        Hydrocarbon type: ``"oil"`` or ``"gas"``.
    vcl : float or np.ndarray, optional
        Clay volume fraction. Enables the preliminary-porosity approach.
    rho_ma : float
        Matrix density (g/cc).
    rho_cl : float
        Clay / shale density (g/cc).
    rho_hc : float, optional
        Hydrocarbon density in the flushed zone (g/cc). Default: 0.8 for oil,
        0.2 for gas.
    floor_porosity : bool, optional
        Count a negative porosity as 0, so that rock denser than its matrix
        (a tight or heavy-mineral bed) gets no correction; a negative
        porosity would lower RHOB there. Default (None): True, and False
        while a benchmark package is reproduced.

    Returns
    -------
    float or np.ndarray
        Hydrocarbon-corrected bulk density (g/cc).
    """
    if rho_hc is None:
        rho_hc = 0.2 if hc_type.lower() == "gas" else 0.8
    if floor_porosity is None:
        floor_porosity = not reproducing_benchmark()

    if vcl is not None:
        numerator = rho_ma - rhob - vcl * (rho_ma - rho_cl)
        denominator = rho_ma - rhob_fluid * sxo - rho_hc * (1.0 - sxo)

        if isinstance(denominator, np.ndarray):
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        else:
            denominator = denominator if abs(denominator) > 1e-10 else 1e-10

        phi_d_prelim = numerator / denominator
        if floor_porosity:
            phi_d_prelim = np.clip(phi_d_prelim, 0.0, None)
        rhob_corrected = rhob + phi_d_prelim * (1.0 - sxo) * (rhob_fluid - rho_hc)
    else:
        if floor_porosity:
            phi = np.clip(phi, 0.0, None)
        rhob_corrected = rhob + phi * (1.0 - sxo) * (rhob_fluid - rho_hc)

    return rhob_corrected
