"""
Waxman-Smits equation for water saturation in shaly clastics with CEC data.

This module implements the Waxman-Smits model which accounts for
clay conductivity through the cation exchange capacity (CEC).
"""

from typing import Union
import numpy as np


def waxman_smits(
    phi_t: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: float,
    cec: float,
    rho_ma: float,
    b_coeff: float,
    m_star: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Waxman-Smits water saturation for shaly sands with CEC data.

    Equations
    ---------
    Conductivity form:
        Ct = (1/F*) * (Cw + B * Qv * Sw) * Sw^n

    Where:
        Ct   = 1/Rt  (formation conductivity)
        Cw   = 1/Rw  (water conductivity)
        F*   = phi_t^(-m*)  (intrinsic formation factor)
        B    = equivalent conductance of clay counter-ions
        Qv   = CEC * rho_ma * (1 - phi_t) / phi_t

    First-order approximation:
        Sw = [F* * Ct / (Cw + B * Qv)]^(1/n)

    Parameters
    ----------
    phi_t : float or np.ndarray
        Total porosity (fraction)
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    cec : float
        Cation exchange capacity (meq/100g)
    rho_ma : float
        Matrix density (g/cc)
    b_coeff : float
        Equivalent conductance of counter-ions
        (temperature-dependent, from charts)
    m_star : float
        Intrinsic cementation exponent (from SCAL, default: 2.0)
    n : float
        Saturation exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Water saturation Sw (fraction, clipped 0-1)
    """
    cw = 1.0 / rw
    ct = 1.0 / rt

    # Protect against zero porosity
    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t) if isinstance(phi_t, np.ndarray) else max(phi_t, 1e-10)

    # Intrinsic formation factor
    f_star = phi_t_safe ** (-m_star)

    # Qv: CEC per unit pore volume
    qv = cec * rho_ma * (1.0 - phi_t_safe) / phi_t_safe

    # First-order solution
    denominator = cw + b_coeff * qv
    # Protect against zero denominator
    if isinstance(denominator, np.ndarray):
        denominator = np.where(denominator <= 0, 1e-10, denominator)
    else:
        denominator = max(denominator, 1e-10)

    sw = (f_star * ct / denominator) ** (1.0 / n)

    return np.clip(sw, 0.0, 1.0)


def waxman_smits_sw(
    phi_t: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: float,
    cec: float,
    rho_ma: float,
    b_coeff: float,
    m_star: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """Alias for waxman_smits function."""
    return waxman_smits(phi_t, rt, rw, cec, rho_ma, b_coeff, m_star, n)
