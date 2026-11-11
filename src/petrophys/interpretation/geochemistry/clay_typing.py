"""Clay and mica typing from the thorium/potassium ratio of spectral gamma ray.

Clay minerals differ in how much potassium and thorium they hold: kaolinite
has almost no potassium, illite and micas a lot, glauconite and feldspar more
potassium than thorium. The Th/K ratio (ppm/%) therefore separates them,
read in bands of the Schlumberger Th-K mineral chart (Quirein et al., 1982;
Fertl, 1979). The bands are indicative: a sample is a mixture, the bands
overlap in the literature (the lower ones most), and clean sands, where both
elements are near zero, give an unstable ratio. Classify shaly samples only
(a potassium floor and, ideally, a VCL cutoff), and pass your own
``thresholds`` when a company chart differs.

References
----------
Quirein, J.A., Gardner, J.S. and Watson, J.T. (1982). Combined natural gamma
ray spectral/litho-density measurements applied to complex lithologies.
SPE 11143.

Fertl, W.H. (1979). Gamma ray spectral data assists in complex formation
evaluation. *The Log Analyst* 20(5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Label of samples left untyped (too little potassium, masked out, or missing data).
UNCLASSIFIED = "unclassified"

#: Th/K (ppm/%) upper bounds of each band, and the band names, low to high.
TH_K_CLAY_TYPES: Tuple[Tuple[float, str], ...] = (
    (0.6, "feldspar"),
    (1.6, "glauconite"),
    (2.0, "mica"),
    (3.5, "illite"),
    (12.0, "mixed_layer"),
    (28.0, "kaolinite"),
    (np.inf, "heavy_thorium_minerals"),
)


@dataclass
class ClayTypingResult:
    """Th/K clay typing of a well or of arrays."""

    th_k_ratio: np.ndarray
    #: Band name per sample, ``"unclassified"`` where the sample was not typed.
    clay_type: np.ndarray
    #: Share of the classified samples in each band.
    fractions: Dict[str, float] = field(default_factory=dict)
    classified: int = 0
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __repr__(self) -> str:
        shares = ", ".join(f"{k} {v:.0%}" for k, v in self.fractions.items() if v >= 0.005)
        return f"ClayTypingResult({self.classified} of {self.clay_type.size} samples typed: {shares})"


def classify_clay_th_k(
    thorium_ppm: Any,
    potassium_percent: Any,
    *,
    min_potassium: float = 0.5,
    mask: Optional[Any] = None,
    thresholds: Sequence[Tuple[float, str]] = TH_K_CLAY_TYPES,
) -> ClayTypingResult:
    """Classify samples by Th/K ratio.

    Parameters
    ----------
    thorium_ppm, potassium_percent : array-like
        Thorium in ppm and potassium in percent (not a fraction).
    min_potassium : float
        Samples with less potassium (%) are left unclassified: the ratio of
        two small numbers is noise. 0.5 % keeps shaly rock.
    mask : array-like of bool, optional
        Classify only these samples, e.g. ``vcl > 0.3``.
    thresholds : sequence of (float, str)
        Th/K bands as (upper bound, band name), low to high;
        :data:`TH_K_CLAY_TYPES` by default.

    Returns
    -------
    ClayTypingResult
        Ratio, band name per sample (``"unclassified"`` where too little
        potassium, outside ``mask`` or missing data) and band shares of the
        classified samples.

    Examples
    --------
    >>> result = classify_clay_th_k([12.0, 20.0, 1.0], [4.0, 1.0, 3.0])
    >>> list(result.clay_type)
    ['illite', 'kaolinite', 'feldspar']
    """
    th = np.asarray(thorium_ppm, dtype=float)
    k = np.asarray(potassium_percent, dtype=float)
    th, k = np.broadcast_arrays(th, k)
    bounds = [float(upper) for upper, _ in thresholds]
    if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
        raise ValueError("thresholds must have increasing upper bounds")
    if any(name == UNCLASSIFIED for _, name in thresholds):
        raise ValueError(f"{UNCLASSIFIED!r} labels untyped samples; give the band another name")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(th) & np.isfinite(k) & (k > 0), th / k, np.nan)
    usable = np.isfinite(ratio) & (k >= min_potassium) & (th >= 0)
    if mask is not None:
        usable &= np.asarray(mask, dtype=bool)
    labels = np.full(ratio.shape, UNCLASSIFIED, dtype=object)
    lower = -np.inf
    for upper, name in thresholds:
        labels[usable & (ratio >= lower) & (ratio < upper)] = name
        lower = upper
    classified = int(usable.sum())
    fractions = {
        name: (float(np.count_nonzero(labels == name)) / classified if classified else 0.0)
        for _, name in thresholds
    }
    return ClayTypingResult(th_k_ratio=ratio, clay_type=labels, fractions=fractions, classified=classified)


def clay_typing_from_well(
    well: Any,
    *,
    min_potassium: float = 0.5,
    vcl_curve: Optional[str] = None,
    vcl_cutoff: float = 0.3,
    thresholds: Sequence[Tuple[float, str]] = TH_K_CLAY_TYPES,
    alias_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    add_curves: bool = False,
) -> ClayTypingResult:
    """Th/K clay typing of a well's spectral gamma ray.

    Thorium and potassium are found and converted to ppm and percent as
    :func:`~petrophys.interpretation.geochemistry.interpret_spectral_gamma`
    does (HTHO/THOR, HFK/POTA, ...). With ``vcl_curve`` only samples with
    VCL at or above ``vcl_cutoff`` are typed. With ``add_curves`` the well
    gets ``TH_K_RATIO`` and ``CLAY_TYPE_THK``, the band number (1 for the
    first band of ``thresholds``), with the band names in the curve
    description.
    """
    from petrophys.interpretation.geochemistry.spectral_gamma import interpret_spectral_gamma

    gamma = interpret_spectral_gamma(well, add_curves=False, alias_overrides=alias_overrides)
    th = gamma.measurements.get("THORIUM")
    k = gamma.measurements.get("POTASSIUM")
    if th is None or k is None:
        missing = [name for name, values in (("thorium", th), ("potassium", k)) if values is None]
        raise ValueError(f"Th/K clay typing needs spectral gamma {' and '.join(missing)}")
    mask = None
    warnings = list(gamma.warnings)
    if vcl_curve is not None:
        curve = well.get(vcl_curve)
        if curve is None:
            raise ValueError(f"No curve {vcl_curve!r} on the well")
        vcl = np.asarray(curve.data, dtype=float)
        mask = np.isfinite(vcl) & (vcl >= vcl_cutoff)
    result = classify_clay_th_k(th, k, min_potassium=min_potassium, mask=mask, thresholds=thresholds)
    result.sources = {"THORIUM": gamma.sources["THORIUM"], "POTASSIUM": gamma.sources["POTASSIUM"]}
    result.warnings = warnings
    if add_curves:
        names = [name for _, name in thresholds]
        codes = np.full(result.clay_type.shape, np.nan)
        for index, name in enumerate(names, start=1):
            codes[result.clay_type == name] = index
        well.add_curve("TH_K_RATIO", result.th_k_ratio, "ppm/%", "Thorium to potassium ratio")
        well.add_curve(
            "CLAY_TYPE_THK", codes, "",
            "Th/K clay type: " + ", ".join(f"{i}={n}" for i, n in enumerate(names, start=1)),
        )
    return result


__all__ = ["TH_K_CLAY_TYPES", "UNCLASSIFIED", "ClayTypingResult", "classify_clay_th_k", "clay_typing_from_well"]
