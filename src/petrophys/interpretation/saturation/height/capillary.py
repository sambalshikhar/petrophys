"""Capillary-pressure primitives for saturation-height modeling.

Field units throughout:
  Pc   -> psi          height -> ft        IFT (sigma) -> dyne/cm
  k    -> mD           phi    -> fraction  contact angle theta -> degrees
  density -> specific gravity (water = 1.0)

Key relations
-------------
Pc above the free-water level (FWL):
    Pc = 0.433 * (SGw - SGhc) * h          [psi], h in ft
Lab -> reservoir Pc conversion:
    Pc_res = Pc_lab * (sigma_res*cos t_res) / (sigma_lab*cos t_lab)
Leverett J-function (dimensionless):
    J = 0.21645 * (Pc / (sigma*cos t)) * sqrt(k/phi)
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Hydrostatic gradient of fresh water, psi/ft (per unit specific gravity).
WATER_GRADIENT_PSI_PER_FT = 0.433
# Leverett J-function unit constant for psi / dyne·cm⁻¹ / mD / fraction.
J_CONST = 0.21645

# Common interfacial-tension / contact-angle systems: (IFT dyne/cm, angle deg).
FLUID_SYSTEMS = {
    "air_mercury": (480.0, 140.0),
    "air_brine": (72.0, 0.0),
    "oil_brine": (30.0, 30.0),
    "gas_brine": (50.0, 0.0),
}

ArrayLike = Union[float, np.ndarray]


def pc_from_height(
    height_ft: ArrayLike,
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
) -> ArrayLike:
    """Capillary pressure (psi) at a height above the free-water level (ft)."""
    if sg_water - sg_hc <= 0:
        raise ValueError("sg_water must exceed sg_hc for a buoyancy gradient")
    h = np.asarray(height_ft, dtype=float)
    pc = WATER_GRADIENT_PSI_PER_FT * (sg_water - sg_hc) * np.clip(h, 0.0, None)
    return float(pc) if pc.ndim == 0 else pc


def height_from_pc(
    pc_psi: ArrayLike,
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
) -> ArrayLike:
    """Height above the free-water level (ft) for a capillary pressure (psi)."""
    delta = WATER_GRADIENT_PSI_PER_FT * (sg_water - sg_hc)
    if delta <= 0:
        raise ValueError("sg_water must exceed sg_hc for a buoyancy gradient")
    h = np.asarray(pc_psi, dtype=float) / delta
    return float(h) if h.ndim == 0 else h


def convert_pc_lab_to_reservoir(
    pc_lab: ArrayLike,
    ift_lab: float,
    theta_lab: float,
    ift_res: float,
    theta_res: float,
) -> ArrayLike:
    """Convert a laboratory Pc to reservoir conditions via IFT and contact angle.

    Uses the magnitude of sigma*cos(theta) so that a positive lab Pc (e.g.
    air-mercury, theta~140 deg) maps to a positive reservoir Pc.
    """
    lab = abs(ift_lab * np.cos(np.radians(theta_lab)))
    res = abs(ift_res * np.cos(np.radians(theta_res)))
    if lab < 1e-9:
        raise ValueError("lab IFT*cos(theta) is ~0; check inputs")
    out = np.asarray(pc_lab, dtype=float) * (res / lab)
    return float(out) if out.ndim == 0 else out


def leverett_j(
    pc_psi: ArrayLike,
    perm_md: ArrayLike,
    phi: ArrayLike,
    ift: float = 30.0,
    theta: float = 30.0,
) -> ArrayLike:
    """Leverett J-function (dimensionless) from Pc, permeability and porosity."""
    pc = np.asarray(pc_psi, dtype=float)
    k = np.asarray(perm_md, dtype=float)
    p = np.asarray(phi, dtype=float)
    st = ift * np.cos(np.radians(theta))
    out = np.full(np.broadcast_shapes(pc.shape, k.shape, p.shape), np.nan)
    pc_b, k_b, p_b = np.broadcast_arrays(pc, k, p)
    ok = np.isfinite(pc_b) & np.isfinite(k_b) & np.isfinite(p_b) & (k_b > 0) & (p_b > 0) & (abs(st) > 1e-9)
    out[ok] = J_CONST * (pc_b[ok] / st) * np.sqrt(k_b[ok] / p_b[ok])
    return float(out) if out.ndim == 0 else out


def pc_from_j(
    j: ArrayLike,
    perm_md: ArrayLike,
    phi: ArrayLike,
    ift: float = 30.0,
    theta: float = 30.0,
) -> ArrayLike:
    """Inverse of :func:`leverett_j`: capillary pressure (psi) from J."""
    jj = np.asarray(j, dtype=float)
    k = np.asarray(perm_md, dtype=float)
    p = np.asarray(phi, dtype=float)
    st = ift * np.cos(np.radians(theta))
    out = np.full(np.broadcast_shapes(jj.shape, k.shape, p.shape), np.nan)
    j_b, k_b, p_b = np.broadcast_arrays(jj, k, p)
    ok = np.isfinite(j_b) & np.isfinite(k_b) & np.isfinite(p_b) & (k_b > 0) & (p_b > 0)
    out[ok] = j_b[ok] * st / (J_CONST * np.sqrt(k_b[ok] / p_b[ok]))
    return float(out) if out.ndim == 0 else out
