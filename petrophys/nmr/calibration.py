"""
NMR permeability calibration against core data.

Provides closed-form fitting of SDR and Timur-Coates constants in log space,
with optional full exponent fitting for Coates via linear regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from petrophys.nmr.nmr_data import NMRData


@dataclass
class NMRCalibration:
    """
    Result of calibrating NMR permeability against core plugs.

    Parameters
    ----------
    model : str
        ``"sdr"`` or ``"coates"``.
    params : dict
        Fitted parameters:
        - SDR: ``{"c": float}``
        - Coates: ``{"c": float, "a": float, "b": float}``
    r2_log : float
        R² in log10 space (1.0 = perfect).
    rmse_log : float
        RMSE in log10 space (decades).
    n_points : int
        Number of core plugs used.
    core_depth : np.ndarray
        Depths of core plugs.
    core_perm : np.ndarray
        Core permeability values (mD).
    nmr_perm : np.ndarray
        NMR permeability at the same depths (mD).

    Examples
    --------
    >>> cal = nmr.calibrate_sdr(core_depth, core_perm)
    >>> print(cal)
    NMRCalibration(model=sdr, n=15, c=6.21, R²_log=0.87, RMSE_log=0.31 dec)
    >>> k = nmr.permeability_sdr(calibration=cal)
    """

    model: str
    params: Dict[str, float]
    r2_log: float
    rmse_log: float
    n_points: int
    core_depth: np.ndarray
    core_perm: np.ndarray
    nmr_perm: np.ndarray

    def __repr__(self) -> str:
        p = self.params
        if self.model == "sdr":
            param_str = f"c={p['c']:.3g}"
        else:
            param_str = f"c={p['c']:.3g}, a={p.get('a', 4):.3g}, b={p.get('b', 2):.3g}"
        return (
            f"NMRCalibration(model={self.model}, n={self.n_points}, "
            f"{param_str}, R²_log={self.r2_log:.3f}, "
            f"RMSE_log={self.rmse_log:.3f} dec)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interp_at_core(
    nmr: "NMRData",
    core_depth: np.ndarray,
    core_perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate NMR log properties to core plug depths and filter valid pairs.

    Returns
    -------
    core_depth_valid, core_perm_valid, nmr_depths_for_interp
        All arrays reduced to rows where both core_perm > 0 and NMR MPHI > 0.
    """
    core_depth = np.asarray(core_depth, dtype=float)
    core_perm = np.asarray(core_perm, dtype=float)
    return core_depth, core_perm


def _interp_curve_at_depth(
    log_depth: np.ndarray,
    log_values: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    """Interpolate a log curve safely for ascending or descending depth."""
    depth = np.asarray(log_depth, dtype=float)
    values = np.asarray(log_values, dtype=float)
    targets = np.asarray(target_depth, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full(targets.shape, np.nan, dtype=float)
    depth = depth[valid]
    values = values[valid]
    order = np.argsort(depth)
    return np.interp(targets, depth[order], values[order], left=np.nan, right=np.nan)


def _log_r2(log_pred: np.ndarray, log_true: np.ndarray) -> float:
    """R² in log10 space."""
    ss_res = np.sum((log_true - log_pred) ** 2)
    ss_tot = np.sum((log_true - np.mean(log_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Public calibration functions
# ---------------------------------------------------------------------------


def calibrate_sdr(
    nmr: "NMRData",
    core_depth: np.ndarray,
    core_perm: np.ndarray,
) -> NMRCalibration:
    """
    Calibrate the SDR constant *c* against core permeability.

    Solves in log10 space (closed-form, no iteration):

    .. math::
        \\log_{10} c = \\overline{
            \\log_{10} k_{core} - 4 \\log_{10} \\phi - 2 \\log_{10} T2_{lm}
        }

    where φ is porosity fraction (MPHI / 100) and T2lm is in ms.

    Parameters
    ----------
    nmr : NMRData
    core_depth : array-like
        Core plug depths (same units as ``nmr.depth``).
    core_perm : array-like
        Core plug air/Klinkenberg permeability (mD).  Must be positive.

    Returns
    -------
    NMRCalibration
    """
    core_depth = np.asarray(core_depth, dtype=float)
    core_perm = np.asarray(core_perm, dtype=float)

    if core_depth.shape != core_perm.shape:
        raise ValueError("core_depth and core_perm must have the same length")

    # Interpolate NMR phi and T2lm to core depths. CMR run files are often
    # descending in depth, so sort before interpolation.
    phi_log = _interp_curve_at_depth(nmr.depth, nmr.mphi / 100.0, core_depth)
    t2lm_log = _interp_curve_at_depth(nmr.depth, nmr.t2_lm(), core_depth)

    # Valid mask: core perm > 0, phi > 0, T2lm > 0
    valid = (core_perm > 0) & (phi_log > 0) & (t2lm_log > 0)
    if valid.sum() < 2:
        raise ValueError(
            f"Need at least 2 valid core-NMR pairs for SDR calibration; "
            f"got {valid.sum()} (check that core depths overlap NMR depth range "
            f"and that MPHI > 0 at those depths)."
        )

    k_c = core_perm[valid]
    phi_v = phi_log[valid]
    t2_v = t2lm_log[valid]
    cd_v = core_depth[valid]

    # Closed-form solution
    log_c = np.mean(np.log10(k_c) - 4.0 * np.log10(phi_v) - 2.0 * np.log10(t2_v))
    c_opt = float(10.0**log_c)

    # Diagnostics
    log_k_nmr = np.log10(c_opt) + 4.0 * np.log10(phi_v) + 2.0 * np.log10(t2_v)
    k_nmr = 10.0**log_k_nmr
    rmse_log = float(np.sqrt(np.mean((log_k_nmr - np.log10(k_c)) ** 2)))
    r2 = _log_r2(log_k_nmr, np.log10(k_c))

    return NMRCalibration(
        model="sdr",
        params={"c": c_opt},
        r2_log=r2,
        rmse_log=rmse_log,
        n_points=int(valid.sum()),
        core_depth=cd_v,
        core_perm=k_c,
        nmr_perm=k_nmr,
    )


def calibrate_coates(
    nmr: "NMRData",
    core_depth: np.ndarray,
    core_perm: np.ndarray,
    fit_exponents: bool = False,
) -> NMRCalibration:
    """
    Calibrate Timur-Coates parameters against core permeability.

    **Fixed exponents** (default, ``fit_exponents=False``)
        Only the constant *c* is fitted (a=4, b=2 fixed).  Closed-form:

        .. math::
            \\log_{10} c = \\frac{1}{4}
            \\left(
                \\log_{10} \\phi + \\frac{b}{a} \\log_{10}\\!\\frac{FFI}{BVI}
                - \\frac{1}{a} \\log_{10} k_{core}
            \\right)

    **Full fit** (``fit_exponents=True``)
        *a*, *b*, and *c* are all fitted via ordinary least squares in log
        space:

        .. math::
            \\log_{10} k = a \\log_{10}\\!\\frac{\\phi}{c} +
                          b \\log_{10}\\!\\frac{FFI}{BVI}

        Rewritten as a linear model: ``[log10(phi), log10(FFI/BVI), 1] · θ``.

    Parameters
    ----------
    nmr : NMRData
    core_depth : array-like
    core_perm : array-like
        Core plug permeability (mD).  Must be positive.
    fit_exponents : bool
        ``False`` (default): fit ``c`` only.
        ``True``: fit ``a``, ``b``, and ``c`` jointly.

    Returns
    -------
    NMRCalibration
    """
    core_depth = np.asarray(core_depth, dtype=float)
    core_perm = np.asarray(core_perm, dtype=float)

    if core_depth.shape != core_perm.shape:
        raise ValueError("core_depth and core_perm must have the same length")

    # Interpolate NMR properties to core depths.
    phi_at_core = _interp_curve_at_depth(nmr.depth, nmr.mphi, core_depth)
    ffi_at_core = _interp_curve_at_depth(nmr.depth, nmr._ffi(), core_depth)
    bvi_at_core = _interp_curve_at_depth(nmr.depth, nmr._bvi(), core_depth)

    # Valid mask
    valid = (core_perm > 0) & (phi_at_core > 0) & (ffi_at_core > 0) & (bvi_at_core > 0)
    if valid.sum() < (3 if fit_exponents else 2):
        raise ValueError(
            f"Need at least {3 if fit_exponents else 2} valid core-NMR pairs "
            f"for Coates calibration; got {valid.sum()}."
        )

    k_c = core_perm[valid]
    phi_v = phi_at_core[valid]
    ffi_v = ffi_at_core[valid]
    bvi_v = bvi_at_core[valid]
    cd_v = core_depth[valid]

    ratio_v = ffi_v / bvi_v

    if not fit_exponents:
        # Fixed a=4, b=2 — closed-form solution for log10(c)
        a, b = 4.0, 2.0
        log_c = np.mean(
            (np.log10(phi_v) + 0.5 * np.log10(ratio_v) - 0.25 * np.log10(k_c))
        )
        c_opt = float(10.0**log_c)
        params = {"c": c_opt, "a": a, "b": b}
    else:
        # Full OLS: log10(k) = a*log10(phi) + b*log10(ffi/bvi) + intercept
        # where intercept = -a*log10(c)
        X = np.column_stack(
            [
                np.log10(phi_v),
                np.log10(ratio_v),
                np.ones(len(k_c)),
            ]
        )
        y = np.log10(k_c)
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a = float(theta[0])
        b = float(theta[1])
        intercept = float(theta[2])
        # intercept = -a * log10(c)  =>  log10(c) = -intercept / a
        c_opt = float(10.0 ** (-intercept / a)) if a != 0 else 10.0
        params = {"c": c_opt, "a": a, "b": b}

    # Compute fitted permeability for diagnostics
    log_k_nmr = params["a"] * np.log10(phi_v / params["c"]) + params["b"] * np.log10(
        ratio_v
    )
    k_nmr = 10.0**log_k_nmr
    rmse_log = float(np.sqrt(np.mean((log_k_nmr - np.log10(k_c)) ** 2)))
    r2 = _log_r2(log_k_nmr, np.log10(k_c))

    return NMRCalibration(
        model="coates",
        params=params,
        r2_log=r2,
        rmse_log=rmse_log,
        n_points=int(valid.sum()),
        core_depth=cd_v,
        core_perm=k_c,
        nmr_perm=k_nmr,
    )
