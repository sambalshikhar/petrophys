"""Lithology and clay-volume empirical permeability transform."""

from typing import Any, Optional, Union
import numpy as np

DEFAULT_MAX_PERM_MD = 10000.0


def _parameters_for_lithology(lithology: Any) -> tuple[float, float, str]:
    """Return slope/intercept for the Dunite-style lithology code."""
    try:
        lith_code = float(lithology)
        if 11.0 <= lith_code <= 15.0:
            return 20.0, -3.0, "limestone"
        if lith_code == 17.0:
            return 18.0, -4.0, "dolomite"
        if lith_code == 24.0:
            return 1.0, -2.0, "anhydrite"
        if lith_code == 20.0:
            return 1.0, -4.0, "coal"
        if lith_code == 16.0:
            return 28.0, -2.0, "calcareous_sand"
        return 28.0, -2.0, "sandstone"
    except (TypeError, ValueError):
        text = str(lithology or "").strip().lower().replace("-", "_").replace(" ", "_")

    if "lime" in text or text in {"carbonate", "sandy_limestone"}:
        return 20.0, -3.0, "limestone"
    if "dolo" in text:
        return 18.0, -4.0, "dolomite"
    if "anhy" in text:
        return 1.0, -2.0, "anhydrite"
    if "coal" in text:
        return 1.0, -4.0, "coal"
    if "calcareous" in text and "sand" in text:
        return 28.0, -2.0, "calcareous_sand"
    return 28.0, -2.0, "sandstone"


def lith_vcl_permeability(
    phit: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    lith: Optional[Union[float, np.ndarray]] = None,
    permvclscalar: float = 1.2,
    max_perm: Optional[float] = DEFAULT_MAX_PERM_MD,
) -> Union[float, np.ndarray]:
    """
    Calculate permeability from total porosity, VCL, and lithology.

    The transform is:
        perm = 10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)

    Values are returned only where ``VCL > 0`` and ``PHIT > 0.001``.
    Unknown lithology falls back to sandstone parameters.
    Finite permeability values are capped to ``max_perm`` when provided.
    """
    phit_arr = np.asarray(phit, dtype=float)
    vcl_arr = np.asarray(vcl, dtype=float)
    phit_b, vcl_b = np.broadcast_arrays(phit_arr, vcl_arr)

    if lith is None:
        lith_b = np.full(phit_b.shape, np.nan, dtype=object)
    else:
        lith_arr = np.asarray(lith, dtype=object)
        lith_b = np.broadcast_to(lith_arr, phit_b.shape)

    slope = np.full(phit_b.shape, 28.0, dtype=float)
    intercept = np.full(phit_b.shape, -2.0, dtype=float)

    for idx, value in np.ndenumerate(lith_b):
        slope[idx], intercept[idx], _ = _parameters_for_lithology(value)

    exponent = slope * phit_b * (1.0 - vcl_b * permvclscalar) + intercept
    valid = np.isfinite(phit_b) & np.isfinite(vcl_b) & (vcl_b > 0.0) & (phit_b > 0.001)

    perm = np.full(phit_b.shape, np.nan, dtype=float)
    perm[valid] = np.power(10.0, np.minimum(4.5, exponent[valid]))
    if max_perm is not None:
        finite = np.isfinite(perm)
        perm[finite] = np.minimum(perm[finite], float(max_perm))

    if np.isscalar(phit) and np.isscalar(vcl) and (lith is None or np.isscalar(lith)):
        return float(perm)
    return perm


def lith_vcl_parameter_labels(lith: Optional[Union[float, np.ndarray]]) -> np.ndarray:
    """Return lithology parameter labels for trace metadata."""
    lith_arr = np.asarray(lith if lith is not None else np.nan, dtype=object)
    labels = np.empty(lith_arr.shape, dtype=object)
    for idx, value in np.ndenumerate(lith_arr):
        labels[idx] = _parameters_for_lithology(value)[2]
    return labels
