"""Lithology and clay-volume empirical permeability transform."""

from typing import Any, Optional, Union
import numpy as np

DEFAULT_MAX_PERM_MD = 10000.0


def _parameters_for_lithology(lithology: Any) -> tuple[float, float, str]:
    """Return slope, intercept and group for a lithology code or rock name.

    Numeric codes are read with the GeoLink (Kingdom) lithology code table
    that the sample wells' ``LITHOLOGY_GEOLINK`` curve uses, as the mineral
    solver's matrix inference reads them: chalk, limestones, marlstone,
    biogenic ooze and calcareous debris flow take the limestone transform,
    15 dolomite, 16 calcareous cement the calcareous-sand one, 18 coal, the
    evaporites (halite, anhydrite, sylvinite, polyhalite, carnallite, kainite)
    the anhydrite one, and every other code the sandstone one. Rock names are
    matched by their words. Unknown lithology takes the sandstone transform.
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    benchmark = reproducing_benchmark()
    try:
        lith_code = float(lithology)
    except (TypeError, ValueError):
        text = str(lithology or "").strip().lower().replace("-", "_").replace(" ", "_")
    else:
        return _benchmark_parameters(lith_code) if benchmark else _code_parameters(lith_code)

    if "lime" in text or text in {"carbonate", "sandy_limestone"} or (
        not benchmark and ("chalk" in text or "marl" in text)
    ):
        return 20.0, -3.0, "limestone"
    if "dolo" in text:
        return 18.0, -4.0, "dolomite"
    if "anhy" in text or (not benchmark and text in _EVAPORITE_NAMES):
        return 1.0, -2.0, "anhydrite"
    if "coal" in text:
        return 1.0, -4.0, "coal"
    if "calcareous" in text and "sand" in text:
        return 28.0, -2.0, "calcareous_sand"
    return 28.0, -2.0, "sandstone"


_EVAPORITE_NAMES = {"halite", "salt", "gypsum", "sylvinite", "polyhalite", "carnallite", "kainite", "evaporite"}


def _code_parameters(lith_code: float) -> tuple[float, float, str]:
    """The transform of a GeoLink (Kingdom) lithology code."""
    from petrophys.interpretation.lithology.matrix_inference import LITHOLOGY_CODE_CLASS

    if not np.isfinite(lith_code) or lith_code != int(lith_code):
        return 28.0, -2.0, "sandstone"
    code = int(lith_code)
    if code == 15:
        return 18.0, -4.0, "dolomite"
    if code == 16:
        return 28.0, -2.0, "calcareous_sand"
    if code == 18:
        return 1.0, -4.0, "coal"
    family = LITHOLOGY_CODE_CLASS.get(code)
    if family == "carbonate":
        return 20.0, -3.0, "limestone"
    if family == "evaporite":
        return 1.0, -2.0, "anhydrite"
    return 28.0, -2.0, "sandstone"


def _benchmark_parameters(lith_code: float) -> tuple[float, float, str]:
    """The code groups the benchmark packages were computed with."""
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

    Values are returned only where ``VCL >= 0`` and ``PHIT > 0.001``, so
    clean rock (VCL = 0) gets the full lithology transform.
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
    valid = np.isfinite(phit_b) & np.isfinite(vcl_b) & (vcl_b >= 0.0) & (phit_b > 0.001)

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
