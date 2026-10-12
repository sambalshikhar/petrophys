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
    min_bin_coverage: float = 1.0,
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
    if bins.shape[1] != t2c.size:
        raise ValueError("bins column count must match t2_centers length")
    if not 0.0 <= min_bin_coverage <= 1.0:
        raise ValueError("min_bin_coverage must be between 0 and 1")
    finite_count = np.count_nonzero(np.isfinite(bins), axis=1)
    coverage = finite_count / max(t2c.size, 1)
    total = np.nansum(bins, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (total > 0) & (coverage >= min_bin_coverage),
            np.exp(np.nansum(bins * ln_t2, axis=1) / total),
            np.nan,
        )


def t2_fraction_below_cutoff(
    t2_centers: np.ndarray,
    cutoff_ms: Array,
) -> np.ndarray:
    """
    Fraction of each T2 bin below the cutoff, interpolated in log T2.

    Each bin spans the log-T2 interval between the geometric means of its
    centre and its neighbours' (the outer bins extend as far beyond their
    centre as inward), so a cutoff inside a bin splits that bin by log-T2
    width and the partition changes continuously with the cutoff. A single
    bin counts whole, by its centre.

    Parameters
    ----------
    t2_centers : np.ndarray
        T2 bin centres (ms), increasing.
    cutoff_ms : float or np.ndarray
        T2 cutoff (ms), or one cutoff per depth.

    Returns
    -------
    np.ndarray
        Shape (n_bins,), or (n_depth, n_bins) for per-depth cutoffs.

    Examples
    --------
    >>> t2 = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
    >>> np.round(t2_fraction_below_cutoff(t2, 33.0), 3)[2:5].tolist()
    [1.0, 0.544, 0.0]
    """
    log_t2 = np.log(np.asarray(t2_centers, dtype=float))
    log_cut = np.log(np.asarray(cutoff_ms, dtype=float))[..., None]
    if log_t2.size < 2:
        return np.where(log_t2 < log_cut, 1.0, 0.0)
    edges = (log_t2[:-1] + log_t2[1:]) / 2.0
    lower = np.concatenate([[2.0 * log_t2[0] - edges[0]], edges])
    upper = np.concatenate([edges, [2.0 * log_t2[-1] - edges[-1]]])
    with np.errstate(invalid="ignore"):
        return np.clip((log_cut - lower) / (upper - lower), 0.0, 1.0)


def bvi_from_bins(
    bins: np.ndarray,
    t2_centers: np.ndarray,
    cutoff_ms: float = 33.0,
    min_bin_coverage: float = 1.0,
    partial_bins: bool = True,
) -> np.ndarray:
    """
    Bulk Volume Irreducible from T2 bins at a given cutoff.

    Sums the bins below ``cutoff_ms``; the bin that straddles the cutoff
    counts with the part of its log-T2 width below it
    (:func:`t2_fraction_below_cutoff`).

    Parameters
    ----------
    bins : np.ndarray
        Shape (n_depth, n_bins).
    t2_centers : np.ndarray
        T2 bin centres (ms).
    cutoff_ms : float
        T2 cutoff (ms).  Default 33 ms, the sandstone value (Coates, Xiao and
        Prammer, 1999).
    partial_bins : bool
        ``False`` counts whole bins with centre **< cutoff_ms**, the split
        some vendor summaries use.

    Returns
    -------
    np.ndarray
        BVI (pu).
    """
    t2c  = np.asarray(t2_centers, dtype=float)
    if partial_bins:
        weights = t2_fraction_below_cutoff(t2c, float(cutoff_ms))
    else:
        weights = (t2c < cutoff_ms).astype(float)
    return _sum_selected_bins(bins, weights, min_bin_coverage)


def ffi_from_bins(
    bins: np.ndarray,
    t2_centers: np.ndarray,
    cutoff_ms: float = 33.0,
    min_bin_coverage: float = 1.0,
    partial_bins: bool = True,
) -> np.ndarray:
    """
    Free Fluid Index from T2 bins at a given cutoff.

    Sums the bins above ``cutoff_ms``, the complement of
    :func:`bvi_from_bins`; ``partial_bins=False`` counts whole bins with
    centre **>= cutoff_ms**.
    """
    t2c  = np.asarray(t2_centers, dtype=float)
    if partial_bins:
        weights = 1.0 - t2_fraction_below_cutoff(t2c, float(cutoff_ms))
    else:
        weights = (t2c >= cutoff_ms).astype(float)
    return _sum_selected_bins(bins, weights, min_bin_coverage)


def _sum_selected_bins(
    bins: np.ndarray,
    weights: np.ndarray,
    min_bin_coverage: float,
) -> np.ndarray:
    """Sum the weighted part of each bin without interpreting missing distributions as zero."""
    values = np.atleast_2d(np.asarray(bins, dtype=float))
    weight = np.asarray(weights, dtype=float)
    if weight.ndim != 1 or values.shape[1] != weight.size:
        raise ValueError("bin mask length must match the number of bin columns")
    if not 0.0 <= min_bin_coverage <= 1.0:
        raise ValueError("min_bin_coverage must be between 0 and 1")
    selected = weight > 0.0
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        return np.zeros(values.shape[0], dtype=float)
    subset = values[:, selected]
    coverage = np.count_nonzero(np.isfinite(subset), axis=1) / selected_count
    result = np.nansum(subset * weight[selected], axis=1)
    result[coverage < min_bin_coverage] = np.nan
    return result
