"""
Standalone NMR permeability functions.

Use these when you already have T2lm, FFI, and BVI as plain arrays
and don't need the full NMRData container.
"""

from __future__ import annotations

import numpy as np
from typing import Union

Array = Union[float, np.ndarray]


def sdr_permeability(
    phi_frac: Array,
    t2lm_ms: Array,
    c: float = 4.0,
) -> np.ndarray:
    """
    SDR (Schlumberger-Doll Research) NMR permeability.

    .. math::
        k = C \\cdot \\phi^{4} \\cdot T2_{lm}^{2}

    Parameters
    ----------
    phi_frac : float or np.ndarray
        NMR porosity as a **fraction** (0–1).
    t2lm_ms : float or np.ndarray
        Logarithmic mean T2 (ms).
    c : float
        Empirical constant (default 4.0 — Kenyon 1992).

    Returns
    -------
    np.ndarray
        Permeability (mD).
    """
    phi  = np.asarray(phi_frac, dtype=float)
    t2lm = np.asarray(t2lm_ms,  dtype=float)
    with np.errstate(invalid="ignore"):
        return c * phi ** 4 * t2lm ** 2


def coates_permeability(
    phi_pu: Array,
    ffi_pu: Array,
    bvi_pu: Array,
    c: float = 10.0,
    a: float = 4.0,
    b: float = 2.0,
) -> np.ndarray:
    """
    Timur-Coates NMR permeability.

    .. math::
        k = \\left(\\frac{\\phi}{C}\\right)^{a}
            \\cdot \\left(\\frac{FFI}{BVI}\\right)^{b}

    Parameters
    ----------
    phi_pu : float or np.ndarray
        NMR porosity (porosity units, pu = percent).
    ffi_pu : float or np.ndarray
        Free Fluid Index (pu).
    bvi_pu : float or np.ndarray
        Bulk Volume Irreducible (pu).
    c : float
        Coates constant (default 10.0).
    a : float
        Porosity exponent (default 4.0).
    b : float
        FFI/BVI exponent (default 2.0).

    Returns
    -------
    np.ndarray
        Permeability (mD).  NaN where BVI = 0.
    """
    phi = np.asarray(phi_pu, dtype=float)
    ffi = np.asarray(ffi_pu, dtype=float)
    bvi = np.asarray(bvi_pu, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bvi > 0, ffi / bvi, np.nan)
        return (phi / c) ** a * ratio ** b


def t2lm_from_bins(
    bins: np.ndarray,
    t2_centers: np.ndarray,
) -> np.ndarray:
    """
    Logarithmic mean T2 from a T2 bin distribution array.

    Parameters
    ----------
    bins : np.ndarray
        Shape (n_depth, n_bins) — porosity (pu) in each T2 bin.
    t2_centers : np.ndarray
        Shape (n_bins,) — T2 centre times (ms).

    Returns
    -------
    np.ndarray
        T2lm (ms), shape (n_depth,).  NaN where total bin porosity is 0.
    """
    bins     = np.atleast_2d(np.asarray(bins,       dtype=float))
    t2c      = np.asarray(t2_centers, dtype=float)
    ln_t2    = np.log(t2c)
    total    = np.nansum(bins, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            total > 0,
            np.exp(np.nansum(bins * ln_t2, axis=1) / total),
            np.nan,
        )


def bvi_from_bins(
    bins: np.ndarray,
    t2_centers: np.ndarray,
    cutoff_ms: float = 32.0,
) -> np.ndarray:
    """
    Bulk Volume Irreducible from T2 bins at a given cutoff.

    Sums all bins with T2 centre **< cutoff_ms**.

    Parameters
    ----------
    bins : np.ndarray
        Shape (n_depth, n_bins).
    t2_centers : np.ndarray
        T2 bin centres (ms).
    cutoff_ms : float
        T2 cutoff (ms).  Default 32 ms (clastics).

    Returns
    -------
    np.ndarray
        BVI (pu).
    """
    t2c  = np.asarray(t2_centers, dtype=float)
    mask = t2c < cutoff_ms
    return np.nansum(np.atleast_2d(bins)[:, mask], axis=1)


def ffi_from_bins(
    bins: np.ndarray,
    t2_centers: np.ndarray,
    cutoff_ms: float = 32.0,
) -> np.ndarray:
    """
    Free Fluid Index from T2 bins at a given cutoff.

    Sums all bins with T2 centre **>= cutoff_ms**.
    """
    t2c  = np.asarray(t2_centers, dtype=float)
    mask = t2c >= cutoff_ms
    return np.nansum(np.atleast_2d(bins)[:, mask], axis=1)
