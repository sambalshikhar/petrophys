"""Thomas-Stieber shale distribution: laminar, dispersed and structural shale.

On a crossplot of total porosity against shale volume, a clean sand
(Vsh = 0, phi_sand) and the pure shale (Vsh = 1, phi_shale) bound the
possible mixtures (Thomas and Stieber, 1975):

- laminar shale (beds of shale between sand beds) moves the point along the
  straight line from sand to shale: ``phi_t = (1 - V_lam) phi_sand + V_lam phi_shale``;
- dispersed shale filling the sand's pores lowers porosity by
  ``(1 - phi_shale)`` per unit shale, down to the point
  ``(phi_sand, phi_sand * phi_shale)`` where the pores are full;
- structural shale replacing sand grains adds its own porosity, above the
  laminar line.

For a point between the laminar and dispersed lines the decomposition is
exact: ``V_lam = (phi_t - phi_sand + V_sh (1 - phi_shale)) / (1 - phi_sand)``
and the dispersed shale fills ``V_disp`` of the sand beds' bulk volume. The
sand beds then have porosity ``phi_sand - V_disp (1 - phi_shale)``, the
porosity to use with their own resistivity (:func:`laminated_sand_resistivity`)
in a laminated-sand evaluation. Points above the laminar line are read as
laminar plus structural shale; points below the dispersed line are outside
the model (flagged, clipped).

Pick ``phi_sand`` from the cleanest, best sands and ``phi_shale`` from the
shales of the interval; Vsh is usually from gamma ray.

References
----------
Thomas, E.C. and Stieber, S.J. (1975). The distribution of shale in
sandstones and its effect upon porosity. SPWLA 16th Annual Logging
Symposium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Regime codes of :class:`ThomasStieberResult`.
REGIMES = {0: "outside", 1: "laminar_dispersed", 2: "laminar_structural"}


@dataclass
class ThomasStieberResult:
    """Shale distribution per sample."""

    v_laminar: np.ndarray
    #: Dispersed shale as a fraction of the sand beds' bulk volume.
    v_dispersed: np.ndarray
    #: Structural shale as a fraction of the sand beds' bulk volume.
    v_structural: np.ndarray
    #: Porosity of the sand beds (between the laminae).
    sand_porosity: np.ndarray
    #: Sand-bed fraction of the interval, ``1 - V_lam``.
    net_to_gross: np.ndarray
    #: 1 laminar + dispersed, 2 laminar + structural, 0 outside the model, NaN missing.
    regime: np.ndarray

    def __repr__(self) -> str:
        if np.ndim(self.regime) == 0:
            return (f"ThomasStieberResult(V_lam {float(self.v_laminar):.3f}, V_disp {float(self.v_dispersed):.3f}, "
                    f"V_str {float(self.v_structural):.3f}, regime {REGIMES.get(int(self.regime), 'missing') if np.isfinite(self.regime) else 'missing'})")
        counts = {REGIMES[k]: int(np.sum(self.regime == k)) for k in REGIMES}
        return f"ThomasStieberResult({np.size(self.regime)} samples, mean N/G {np.nanmean(self.net_to_gross):.2f}, {counts})"


def thomas_stieber(
    phit: Any,
    vsh: Any,
    phi_sand: float,
    phi_shale: float,
) -> ThomasStieberResult:
    """Decompose total porosity and shale volume into laminar, dispersed and structural shale.

    Parameters
    ----------
    phit : array-like
        Total porosity (v/v).
    vsh : array-like
        Shale volume (v/v), e.g. from gamma ray.
    phi_sand : float
        Porosity of the clean sand end point.
    phi_shale : float
        Total porosity of the pure shale.

    Examples
    --------
    Half laminar shale, clean sand beds:

    >>> r = thomas_stieber(0.5 * 0.25 + 0.5 * 0.10, 0.5, 0.25, 0.10)
    >>> round(float(r.v_laminar), 3), round(float(r.v_dispersed), 3)
    (0.5, 0.0)
    """
    if not 0 < phi_shale < phi_sand < 1:
        raise ValueError("Need 0 < phi_shale < phi_sand < 1")
    p = np.asarray(phit, dtype=float)
    v = np.asarray(vsh, dtype=float)
    p, v = np.broadcast_arrays(p, v)
    valid = np.isfinite(p) & np.isfinite(v)
    v_c = np.clip(v, 0.0, 1.0)
    laminar_line = phi_sand + v_c * (phi_shale - phi_sand)
    shape = p.shape
    v_lam = np.full(shape, np.nan)
    v_disp = np.full(shape, np.nan)
    v_str = np.full(shape, np.nan)
    sand_phi = np.full(shape, np.nan)
    regime = np.full(shape, np.nan)

    # Below (or on) the laminar line: laminar + dispersed.
    below = valid & (p <= laminar_line + 1e-12)
    lam = (p - phi_sand + v_c * (1.0 - phi_shale)) / (1.0 - phi_sand)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(lam < 1.0, (v_c - lam) / (1.0 - lam), 0.0)
    inside = below & (lam >= -1e-9) & (disp <= phi_sand + 1e-9)
    lam_c = np.clip(lam, 0.0, 1.0)
    disp_c = np.clip(disp, 0.0, phi_sand)
    v_lam[below] = lam_c[below]
    v_disp[below] = disp_c[below]
    v_str[below] = 0.0
    sand_phi[below] = phi_sand - disp_c[below] * (1.0 - phi_shale)
    regime[below] = np.where(inside[below], 1.0, 0.0)

    # Above the laminar line: laminar + structural (shale replacing grains).
    # Sand beds with structural shale V_str have porosity phi_sand + V_str phi_shale
    # and shale content V_str: V_sh = V_lam + (1 - V_lam) V_str,
    # phi_t = V_lam phi_shale + (1 - V_lam)(phi_sand + V_str phi_shale).
    above = valid & ~below
    # Eliminate V_str: phi_t = phi_sand + V_sh phi_shale - V_lam phi_sand
    lam_s = (phi_sand + v_c * phi_shale - p) / phi_sand
    with np.errstate(divide="ignore", invalid="ignore"):
        struct = np.where(lam_s < 1.0, (v_c - lam_s) / (1.0 - lam_s), 0.0)
    inside_s = above & (lam_s >= -1e-9) & (struct <= 1.0 - phi_sand + 1e-9)
    lam_sc = np.clip(lam_s, 0.0, 1.0)
    struct_c = np.clip(struct, 0.0, 1.0 - phi_sand)
    v_lam[above] = lam_sc[above]
    v_disp[above] = 0.0
    v_str[above] = struct_c[above]
    sand_phi[above] = phi_sand + struct_c[above] * phi_shale
    regime[above] = np.where(inside_s[above], 2.0, 0.0)

    def out(values):
        return float(values) if np.ndim(values) == 0 else values

    return ThomasStieberResult(
        v_laminar=out(v_lam), v_dispersed=out(v_disp), v_structural=out(v_str),
        sand_porosity=out(sand_phi), net_to_gross=out(1.0 - v_lam), regime=out(regime),
    )


def laminated_sand_resistivity(rt: Any, v_laminar: Any, r_shale: float) -> np.ndarray:
    """Resistivity of the sand beds of a laminated sequence (parallel conduction).

    Horizontal current through sand and shale laminae:
    ``1/Rt = (1 - V_lam)/R_sand + V_lam/R_shale``. NaN where the shale alone
    would conduct more than measured (``V_lam/R_shale >= 1/Rt``).
    """
    rt_arr = np.asarray(rt, dtype=float)
    lam = np.asarray(v_laminar, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sand_conductance = 1.0 / rt_arr - lam / r_shale
        rsand = np.where(sand_conductance > 0, (1.0 - lam) / sand_conductance, np.nan)
    return float(rsand) if np.ndim(rsand) == 0 else rsand


__all__ = ["REGIMES", "ThomasStieberResult", "laminated_sand_resistivity", "thomas_stieber"]
