"""Rock-physics primitives for sonic (acoustic) water-saturation estimation.

All equations are standard, public-domain rock physics (Gassmann 1951; Wood/Reuss
fluid mixing; Krief and critical-porosity dry-frame models; Greenberg-Castagna
Vs-from-Vp). This module is a clean-room implementation from the published
literature and is deliberately independent of any specific vendor algorithm.

Units: slowness in us/ft, velocity in m/s, density in g/cc, moduli in GPa.
  K [GPa] = density[g/cc] * V[m/s]^2 / 1e6

References
----------
Gassmann, F. (1951). Über die Elastizität poröser Medien.
*Vierteljahrsschrift der Naturforschenden Gesellschaft in Zürich* 96, 1-23.

Greenberg, M.L. and Castagna, J.P. (1992). Shear-wave velocity estimation
in porous rocks: theoretical formulation, preliminary verification and
applications. *Geophysical Prospecting* 40, 195-209.

Krief, M., Garat, J., Stellingwerff, J. and Ventre, J. (1990). A
petrophysical interpretation using the velocities of P and S waves
(full-waveform sonic). *The Log Analyst* 31(6).

Mavko, G., Mukerji, T. and Dvorkin, J. (2009). *The Rock Physics
Handbook*, 2nd ed. Cambridge University Press.

Nur, A., Mavko, G., Dvorkin, J. and Galmudi, D. (1998). Critical porosity:
a key to relating physical properties to porosity in rocks. *The Leading
Edge* 17(3), 357-362.

Reuss, A. (1929). Berechnung der Fließgrenze von Mischkristallen auf Grund
der Plastizitätsbedingung für Einkristalle. *Zeitschrift für Angewandte
Mathematik und Mechanik* 9, 49-58.

Wood, A.B. (1955). *A Textbook of Sound*, 3rd ed.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# Mineral moduli (GPa) and grain density (g/cc) -- Mavko/Mukerji/Dvorkin.
MINERAL_ELASTIC: Dict[str, Dict[str, float]] = {
    "quartz":    {"k": 37.0, "mu": 44.0, "rho": 2.65},
    "calcite":   {"k": 76.8, "mu": 32.0, "rho": 2.71},
    "dolomite":  {"k": 94.9, "mu": 45.0, "rho": 2.87},
    "clay":      {"k": 21.0, "mu":  7.0, "rho": 2.60},
    "anhydrite": {"k": 56.1, "mu": 29.1, "rho": 2.96},
}

# Fluid bulk modulus (GPa) and density (g/cc).
FLUID_ELASTIC: Dict[str, Dict[str, float]] = {
    "brine": {"k": 2.8,  "rho": 1.05},
    "oil":   {"k": 1.0,  "rho": 0.80},
    "gas":   {"k": 0.06, "rho": 0.20},
}

# Greenberg-Castagna Vs = f(Vp) coefficients (Vp, Vs in km/s), brine-saturated.
_GC = {
    "sandstone": (0.0, 0.80416, -0.85588),
    "limestone": (-0.05508, 1.01677, -1.03049),
    "dolomite":  (0.0, 0.58321, -0.07775),
    "shale":     (0.0, 0.76969, -0.86735),
}


def slowness_to_velocity(dt_us_ft: ArrayLike) -> ArrayLike:
    """Convert sonic slowness (us/ft) to velocity (m/s)."""
    dt = np.asarray(dt_us_ft, dtype=float)
    v = np.where(dt > 0, 304800.0 / dt, np.nan)
    return float(v) if np.ndim(dt_us_ft) == 0 else v


def vs_from_vp_greenberg_castagna(vp_ms: ArrayLike, lithology: str = "sandstone") -> ArrayLike:
    """Estimate shear velocity (m/s) from Vp (m/s) via Greenberg-Castagna.

    Used only when a measured DTS is unavailable; it assumes brine saturation,
    so the resulting Sw is approximate (flagged by the caller)."""
    a2, a1, a0 = _GC.get(lithology, _GC["sandstone"])
    vp_km = np.asarray(vp_ms, dtype=float) / 1000.0
    vs_km = a2 * vp_km ** 2 + a1 * vp_km + a0
    vs = np.clip(vs_km, 0.0, None) * 1000.0
    return float(vs) if np.ndim(vp_ms) == 0 else vs


def velocity_to_moduli(vp: ArrayLike, vs: ArrayLike, rhob: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Saturated bulk and shear moduli (GPa) from Vp, Vs (m/s) and density (g/cc)."""
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rhob, dtype=float)
    mu = rho * vs ** 2 / 1e6
    k = rho * (vp ** 2 - 4.0 / 3.0 * vs ** 2) / 1e6
    return k, mu


def dry_rock_modulus(phi: ArrayLike, k_min: ArrayLike, mu_min: float, model: str = "krief",
                     phi_c: float = 0.4) -> Tuple[ArrayLike, ArrayLike]:
    """Dry-frame bulk and shear moduli (GPa) vs porosity.

    ``k_min`` and ``mu_min`` are the mineral bulk and shear moduli (GPa);
    ``k_min`` is one value or one per sample.

    ``krief``  : K_dry = K_min*(1-phi)^(3/(1-phi))  (Krief et al., 1990)
    ``critical_porosity`` : K_dry = K_min*(1 - phi/phi_c) for phi<phi_c (Nur).
    """
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.999)
    if model == "critical_porosity":
        f = np.clip(1.0 - phi / phi_c, 0.0, 1.0)
    else:  # krief
        f = (1.0 - phi) ** (3.0 / np.maximum(1.0 - phi, 1e-6))
    return k_min * f, mu_min * f


def gassmann_ksat(k_dry: ArrayLike, k_min: float, k_fl: ArrayLike, phi: ArrayLike) -> ArrayLike:
    """Gassmann saturated bulk modulus (GPa) from dry frame, mineral, fluid, porosity."""
    k_dry = np.asarray(k_dry, dtype=float)
    k_fl = np.asarray(k_fl, dtype=float)
    phi = np.clip(np.asarray(phi, dtype=float), 1e-4, 0.999)
    num = (1.0 - k_dry / k_min) ** 2
    den = phi / k_fl + (1.0 - phi) / k_min - k_dry / k_min ** 2
    return k_dry + np.where(np.abs(den) > 1e-12, num / den, 0.0)


def gassmann_kfl(k_sat: ArrayLike, k_dry: ArrayLike, k_min: float, phi: ArrayLike) -> ArrayLike:
    """Invert Gassmann for the in-situ fluid modulus K_fl (GPa) given K_sat & K_dry."""
    k_sat = np.asarray(k_sat, dtype=float)
    k_dry = np.asarray(k_dry, dtype=float)
    phi = np.clip(np.asarray(phi, dtype=float), 1e-4, 0.999)
    d_k = k_sat - k_dry
    b = (1.0 - k_dry / k_min) ** 2
    # phi/K_fl = B/dK - (1-phi)/K_min + K_dry/K_min^2
    inv_kfl_term = np.where(np.abs(d_k) > 1e-9, b / d_k, np.inf) - (1.0 - phi) / k_min + k_dry / k_min ** 2
    k_fl = np.where(inv_kfl_term > 1e-9, phi / inv_kfl_term, np.nan)
    return k_fl


def wood_fluid_modulus(sw: ArrayLike, k_brine: float, k_hc: float) -> ArrayLike:
    """Effective fluid bulk modulus (GPa) for a brine/HC mix (Wood/Reuss average)."""
    k_brine, k_hc = _check_fluid_moduli(k_brine, k_hc)
    sw = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
    return 1.0 / (sw / k_brine + (1.0 - sw) / k_hc)


def sw_from_fluid_modulus(k_fl: ArrayLike, k_brine: float, k_hc: float) -> ArrayLike:
    """Water saturation from the in-situ fluid modulus (inverse of Wood mixing)."""
    from petrophys._internal.validation import warn_range

    k_brine, k_hc = _check_fluid_moduli(k_brine, k_hc, distinct=True)
    k_fl = np.asarray(k_fl, dtype=float)
    warn_range(k_fl, "Sonic fluid bulk modulus (GPa)", 0, open_lower=True,
               action="Values are retained and saturation is clipped to 0-1; review the fluid modulus.")
    sw = (1.0 / k_fl - 1.0 / k_hc) / (1.0 / k_brine - 1.0 / k_hc)
    return np.clip(sw, 0.0, 1.0)


def _check_fluid_moduli(k_brine, k_hc, *, distinct=False):
    from petrophys._internal.validation import broadcast_inputs

    k_brine, k_hc = broadcast_inputs(k_brine=k_brine, k_hc=k_hc)
    for name, value in (("k_brine", k_brine), ("k_hc", k_hc)):
        if np.any(~np.isfinite(value) | (value <= 0)):
            raise ValueError(f"{name} must be a finite positive fluid bulk modulus (GPa)")
    if distinct and np.any(k_brine == k_hc):
        raise ValueError("k_brine and k_hc must differ to infer water saturation from fluid modulus")
    return tuple(float(value) if value.ndim == 0 else value for value in (k_brine, k_hc))


def fluid_substitution_dtc_swing(
    k_dry: ArrayLike, k_min: ArrayLike, mu: ArrayLike, phi: ArrayLike,
    rhob: ArrayLike, k_brine: float, k_hc: float,
) -> ArrayLike:
    """Compressional slowness swing (us/ft) between full brine and full HC.

    This is the TOTAL fluid signal a sonic-Sw inversion has to work with at a
    given porosity and frame: the change in DTC if the rock went from 100 %
    water to 100 % hydrocarbon (density held at the measured value, so this
    isolates the Gassmann fluid-modulus effect). When this swing is comparable
    to or smaller than the sonic measurement noise, Sw is not resolvable from
    the sonic regardless of how the inversion is implemented -- which is exactly
    why the method is unreliable in stiff, low-porosity rock.
    """
    k_b = gassmann_ksat(k_dry, k_min, k_brine, phi)
    k_h = gassmann_ksat(k_dry, k_min, k_hc, phi)
    mu = np.asarray(mu, dtype=float)
    rho = np.asarray(rhob, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_b = k_b + 4.0 / 3.0 * mu
        m_h = k_h + 4.0 / 3.0 * mu
        vp_b = np.sqrt(np.where(m_b > 0, m_b, np.nan) / rho * 1e6)
        vp_h = np.sqrt(np.where(m_h > 0, m_h, np.nan) / rho * 1e6)
        dtc_b = 304800.0 / vp_b
        dtc_h = 304800.0 / vp_h
    return np.abs(dtc_h - dtc_b)
