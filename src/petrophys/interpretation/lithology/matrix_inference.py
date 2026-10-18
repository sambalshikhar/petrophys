"""Resolve a well's lithology / matrix when it isn't specified (``"auto"``).

Reliable sources, in priority order (single-curve PEF/N-D heuristics are not
used: on real data a single bad PEF curve gives confidently wrong answers, while
the two sources below agree with vendor ground truth):

  0. Vendor lithology log  -- a per-sample lithology-code curve in the LAS
       (e.g. LITHOLOGY_GEOLINK; see :func:`lithology_code_curve`), decoded via
       LITHOLOGY_CODE_CLASS. Read up front, so it can drive the N-D VCL clean
       line, porosity matrix, Sw routing and net-pay cutoffs for the whole run.
  1. Core matrix density -- the median core grain density (GDEN/CGD) when the
       well carries one; otherwise the apparent matrix density from CORE porosity
       (CPOR/PHIC), rho_ma = (RHOB - phi*rho_fl)/(1 - phi), when 80% or more of
       the samples agree. Knowing the true porosity pins the matrix and breaks
       the quartz/calcite degeneracy WITHOUT PEF. Also a raw-input source, read
       up front.
  2. Multi-mineral solver  -- the VQUAR/VCALC/VDOLO/VANHY mineral volumes the
       LITHOLOGY step computes. Reliable WITH PEF; without PEF it cannot separate
       quartz from calcite, so it is the last data-driven resort. Available only
       after that step, so it feeds the later steps (saturation, net pay).
  3. default -> sandstone / clastic (with a warning).

Explicit ``matrix_type`` / ``lithology`` always win; this is the fallback for
the unspecified case only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise


# matrix -> lithology family used by Sw routing / net-pay cutoffs
_MATRIX_TO_LITHOLOGY = {
    "sandstone": "clastic",
    "limestone": "carbonate",
    "dolomite": "carbonate",
}

# Lithology-code -> coarse class, for the GeoLink / Kingdom nomenclature
# (Litho_Nomenclature_Kingdom.xlsx). "clastic"/"carbonate" drive the decision;
# "dolomite" is split out so a dolomite-dominated carbonate maps to that matrix.
LITHOLOGY_CODE_CLASS: Dict[int, str] = {
    1: "clastic", 2: "clastic", 3: "clastic", 4: "clastic", 5: "clastic",
    6: "clastic", 7: "clastic", 8: "clastic",            # sand/silt/shale
    9: "carbonate", 10: "carbonate", 11: "carbonate",    # chalk/limestone
    12: "carbonate", 13: "carbonate",                    # arg. lst / marlstone
    14: "clastic",                                       # calcareous shale
    15: "dolomite",                                      # dolomite
    16: "carbonate",                                     # calcareous cement
    17: "clastic",                                       # conglomerate
    18: "other", 19: "other", 20: "other",               # coal / volcanic
    21: "evaporite", 22: "evaporite",
    23: "other", 24: "other",
    25: "carbonate",                                     # biogenic ooze
    26: "other",
    27: "evaporite",
    28: "other", 29: "other", 30: "other",
    31: "carbonate",                                     # calcareous debris flow
    32: "evaporite", 33: "evaporite", 34: "evaporite",
    35: "clastic", 36: "clastic",                        # aeolian sst / arkose
}

#: The GeoLink/Kingdom lithology code curve.
GEOLINK_LITHOLOGY_CURVE = "LITHOLOGY_GEOLINK"

#: Key of the ``_geology_context`` well parameter under which the dynamic
#: workflow records the lithology code curve it read (None when it found none).
CODE_CURVE_RECORD = "lithology_code_curve"

# Known curve names that hold a vendor lithology-code log. The detector also
# falls back to a name pattern (*LITH*/*FACIES*) validated as a code log, so this
# list does not need to be exhaustive.
_VENDOR_LITHOLOGY_ALIASES = (
    "LITHOLOGY_GEOLINK", "LITHOLOGY", "LITHOFACIES", "FACIES",
    "LITH_CODE", "LITHO_CODE", "LITHCODE", "LITHOCODE", "LITHOLOGY_CODE",
    "LITHTYPE", "LITHOTYPE", "LITH_TYPE", "ROCK_TYPE", "ROCKTYPE",
    "LITHCLASS", "LITH_CLASS", "FACIES_CODE", "FACIESCODE", "LITHFAC",
    "LITHOLOG", "GEOL_LITH", "LITHOLOGY_INTERP",
)

# Curves the workflow itself produces (different code scheme) — never treat
# these as a vendor log.
_COMPUTED_LITHOLOGY_CURVES = {"LITHO", "MMQC", "LITHO_COMPUTED", "LITHOLOGY_COMPUTED", "LITHO_CLASS"}

# Substrings that look lithology-ish but are not a lithology-code log.
_LITHOLOGY_NAME_DENY = ("LITHOSTAT", "LITHOSTATIC", "FLAG", "PRESS")

# core (lab) porosity curves — ground-truth porosity that pins apparent matrix
# density. PHIE_CC is a core-corrected log porosity (COSTA's is the renamed
# upstream PHIE), not a core measurement; the benchmark context profiles also
# read it, as the packages did.
_CORE_POROSITY_CURVES = (
    "PHIT_CC", "CPOR", "CPOR_CC", "PHIC", "COREPHI", "CPORE",
)
_BENCHMARK_CORE_POROSITY_CURVES = ("PHIE_CC", *_CORE_POROSITY_CURVES)

# core (lab) grain density curves — the measured matrix density
_CORE_GRAIN_DENSITY_CURVES = ("GDEN", "CGD", "GRAIN_DENSITY", "CORE_GDEN")

# matrix grain densities for the apparent-matrix-density classification
_MATRIX_DENSITY_REF = {"sandstone": 2.65, "limestone": 2.71, "dolomite": 2.87}

_MIN_SAMPLES = 20
#: Core grain densities needed for a median; plugs are sparser than logs.
_MIN_GRAIN_DENSITY_SAMPLES = 5
#: Share of the apparent-matrix-density votes a decision needs.
_MIN_CORE_MATRIX_CONFIDENCE = 0.8


@dataclass
class LithologyResolution:
    """Resolved well lithology/matrix and how it was determined."""

    lithology: str               # "clastic" | "carbonate"
    matrix: str                  # "sandstone" | "limestone" | "dolomite"
    method: str                  # "vendor_log" | "multi_mineral_solver" | "default"
    n: int = 0                   # samples backing the decision
    confidence: float = 0.0      # share of (clastic+carbonate) samples for winner
    votes: Dict[str, int] = field(default_factory=dict)


def _clean_mask(well, gr_percentile: float):
    """Boolean mask of low-GR (shale-free) samples; None if GR unavailable."""
    gr, _, _ = get_curve_data_depthwise(well, ["GR"])
    if gr is None:
        return None
    gr = np.asarray(gr, dtype=float)
    valid = np.isfinite(gr)
    if not np.any(valid):
        return None
    threshold = float(np.percentile(gr[valid], gr_percentile))
    return valid & (gr <= threshold)


def _nearest(value: float, ref: Dict[str, float]) -> str:
    """Return the key of ``ref`` whose value is closest to ``value``."""
    return min(ref, key=lambda k: abs(value - ref[k]))


def _finalize(clastic_n, carb_n, dol_n, method, n, warn) -> LithologyResolution:
    """Pick lithology/matrix from clastic vs carbonate (and dolomite) tallies."""
    carbonate_total = carb_n + dol_n
    decided = clastic_n + carbonate_total
    if decided == 0:
        return _default(warn)
    if clastic_n >= carbonate_total:
        lithology, matrix = "clastic", "sandstone"
        conf = clastic_n / decided
    else:
        lithology = "carbonate"
        matrix = "dolomite" if dol_n > carb_n else "limestone"
        conf = carbonate_total / decided
    votes = {"clastic": int(clastic_n), "limestone": int(carb_n), "dolomite": int(dol_n)}
    if warn:
        warnings.warn(
            f"lithology/matrix='auto': resolved '{lithology}'/'{matrix}' via {method} "
            f"({conf:.0%} of {decided} classified samples).",
            stacklevel=3,
        )
    return LithologyResolution(lithology, matrix, method, int(n), float(conf), votes)


def _default(warn) -> LithologyResolution:
    if warn:
        warnings.warn(
            "lithology/matrix='auto' but no vendor lithology log or solver minerals "
            "available; assuming sandstone/clastic.",
            stacklevel=3,
        )
    return LithologyResolution("clastic", "sandstone", "default", 0, 0.0, {})


def _validate_lithology_codes(well, name) -> Optional[np.ndarray]:
    """Return integer lithology codes if ``name`` looks like a decodable code log."""
    curve = well.get(name)
    if curve is None or getattr(curve, "data", None) is None:
        return None
    arr = np.asarray(curve.data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < _MIN_SAMPLES:
        return None
    # must be (near-)integer codes in a plausible range with few distinct values
    if not np.allclose(finite, np.round(finite), atol=1e-6):
        return None
    codes = np.round(finite).astype(int)
    distinct = np.unique(codes)
    if distinct.size > 50 or distinct.min() < 0 or distinct.max() > 99:
        return None
    # at least half the samples must decode to a known lithology class
    known = np.count_nonzero([int(c) in LITHOLOGY_CODE_CLASS for c in codes])
    if known < 0.5 * codes.size:
        return None
    return codes


def _find_vendor_lithology_codes(well) -> Optional[np.ndarray]:
    """Robustly locate a vendor lithology-code log: explicit aliases first, then a
    name-pattern fallback (*LITH*/*FACIES*), each validated as a code log."""
    names = list(getattr(well, "curves", []) or [])
    by_upper = {}
    for n in names:
        by_upper.setdefault(str(n).upper(), n)

    # 1) explicit aliases (in priority order)
    for alias in _VENDOR_LITHOLOGY_ALIASES:
        if alias in by_upper:
            codes = _validate_lithology_codes(well, by_upper[alias])
            if codes is not None:
                return codes

    # 2) name-pattern fallback
    for up, orig in by_upper.items():
        if up in _COMPUTED_LITHOLOGY_CURVES:
            continue
        if any(bad in up for bad in _LITHOLOGY_NAME_DENY):
            continue
        if "LITH" in up or "FACIES" in up:
            codes = _validate_lithology_codes(well, orig)
            if codes is not None:
                return codes
    return None


def _finite_curve(well, name: str):
    """The curve ``name`` of ``well`` when it has a finite sample, else None."""
    curve = well.get(name)
    if curve is None or getattr(curve, "data", None) is None:
        return None
    try:
        data = np.asarray(curve.data, dtype=float)
    except (TypeError, ValueError):
        return None
    return curve if bool(np.any(np.isfinite(data))) else None


def lithology_code_curve(
    well, explicit: Optional[str] = None, allow_aliases: bool = True,
) -> Optional[Tuple[str, Any]]:
    """Find the per-sample lithology code curve of a well.

    The curve is, in this order:

    1. ``explicit``, a curve the caller names, when it has a finite sample;
    2. ``LITHOLOGY_GEOLINK`` with a finite sample, read as it stands;
    3. with ``allow_aliases``, the first other lithology-code mnemonic
       (``LITHOLOGY``, ``LITHOFACIES``, ``FACIES``, ``LITH_CODE``, ...) whose
       curve holds lithology codes: 20 or more finite samples, whole numbers
       from 0 to 99 with at most 50 distinct values, half or more of the
       samples GeoLink codes.

    No other name pattern is tried, so the workflow's computed ``LITHO`` and
    ``LITHO_CLASS`` and the ``GT_*`` group curves are never found. Whatever
    the curve is called, its codes are decoded with the GeoLink/Kingdom code
    table (:data:`LITHOLOGY_CODE_CLASS`).

    Returns
    -------
    tuple or None
        ``(mnemonic, curve)``, the mnemonic in capitals as the well keys it,
        or None when no curve qualifies.
    """
    if explicit is not None and str(explicit).strip():
        name = str(explicit).strip().upper()
        curve = _finite_curve(well, name)
        if curve is not None:
            return name, curve
    curve = _finite_curve(well, GEOLINK_LITHOLOGY_CURVE)
    if curve is not None:
        return GEOLINK_LITHOLOGY_CURVE, curve
    if allow_aliases:
        for alias in _VENDOR_LITHOLOGY_ALIASES:
            if alias == GEOLINK_LITHOLOGY_CURVE:
                continue
            curve = well.get(alias)
            if curve is not None and _validate_lithology_codes(well, alias) is not None:
                return alias, curve
    return None


def _recorded_code_curve(well) -> Tuple[bool, Optional[str]]:
    """Whether the workflow recorded ``well``'s lithology code curve, and its mnemonic."""
    get_parameter = getattr(well, "get_parameter", None)
    context = get_parameter("_geology_context") if callable(get_parameter) else None
    if not isinstance(context, dict) or CODE_CURVE_RECORD not in context:
        return False, None
    name = context[CODE_CURVE_RECORD]
    return True, (str(name) if name else None)


def interpreted_lithology_code_curve(well) -> str:
    """Mnemonic of the lithology code curve the interpretation of ``well`` read.

    The dynamic workflow records its curve (:func:`lithology_code_curve`) on
    the wells it interprets under the ``"current"`` calculation profile, or
    with ``lithology_curve`` set, and slices of the well keep the record.
    Without a record, or when the run found no code curve, this is
    ``LITHOLOGY_GEOLINK``, the only curve the "geolink" profile reads.
    """
    return _recorded_code_curve(well)[1] or GEOLINK_LITHOLOGY_CURVE


def find_lithology_code_curve(well) -> Optional[str]:
    """Mnemonic of the lithology code curve to read on ``well``, or None.

    The curve the interpretation of the well recorded, when it carries a
    record (see :func:`interpreted_lithology_code_curve`); otherwise
    :func:`lithology_code_curve` with the lithology-code mnemonics.
    """
    recorded, name = _recorded_code_curve(well)
    if recorded:
        return name
    found = lithology_code_curve(well)
    return found[0] if found else None


def resolve_from_lithology_log(
    well, warn: bool = True, code_curve: Optional[str] = None,
) -> Optional[LithologyResolution]:
    """Tier 0: classify from a vendor per-sample lithology-code log, if present.

    ``code_curve`` names the curve to classify. Without it, a well the dynamic
    workflow interpreted is classified from the code curve the run recorded
    (none when it found none), and any other well from the first curve that
    holds lithology codes among the lithology-code mnemonics and then any name
    containing LITH or FACIES.
    """
    if code_curve is None:
        recorded, code_curve = _recorded_code_curve(well)
        if recorded and code_curve is None:
            return None
    if code_curve is None:
        codes = _find_vendor_lithology_codes(well)
    else:
        codes = _validate_lithology_codes(well, code_curve)
    if codes is None or codes.size < _MIN_SAMPLES:
        return None
    classes = [LITHOLOGY_CODE_CLASS.get(int(c), "other") for c in codes]
    counts = Counter(classes)
    clastic_n = counts.get("clastic", 0)
    carb_n = counts.get("carbonate", 0)
    dol_n = counts.get("dolomite", 0)
    if clastic_n + carb_n + dol_n < _MIN_SAMPLES:
        return None
    return _finalize(clastic_n, carb_n, dol_n, "vendor_log", codes.size, warn)


def resolve_from_core_matrix(
    well, rho_fluid: float = 1.0, rho_hc: float = 0.8,
    gr_percentile: float = 60.0, warn: bool = True,
) -> Optional[LithologyResolution]:
    """Classify from the core matrix density, if the well carries core data.

    A core grain density curve (GDEN, CGD) measures the matrix density, so
    its median decides the matrix. Without one, the true (core) porosity pins
    the apparent matrix density:
        rho_ma = (RHOB - phi*rho_fl_eff) / (1 - phi)
    which breaks the quartz/calcite degeneracy without PEF or a lithology log.

    When a water-saturation curve is present, the pore fluid density is
    HC-corrected per sample, ``rho_fl_eff = Sw*rho_fluid + (1-Sw)*rho_hc``;
    otherwise water is assumed. This matters in hydrocarbon-bearing reservoir
    with no water-leg reference, where assuming water alone biases apparent
    matrix density low and a carbonate can read as quartz.

    Classification is done over clean (low-GR) samples. The nearest-matrix
    vote decides only when at least 80% of the samples agree; otherwise None
    is returned and the later sources (solver, default) decide. Inside
    ``historical_benchmark()`` the vote is ungated, PHIE_CC counts as core
    porosity and the grain density is not read, as the benchmark packages
    were computed.
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    historical = reproducing_benchmark()
    if not historical:
        gden_c, _, _ = get_curve_data_depthwise(well, list(_CORE_GRAIN_DENSITY_CURVES))
        if gden_c is not None:
            gden = np.asarray(gden_c, dtype=float)
            gden = gden[np.isfinite(gden) & (gden > 2.0) & (gden < 3.2)]
            if gden.size >= _MIN_GRAIN_DENSITY_SAMPLES:
                median = float(np.median(gden))
                matrix = _nearest(median, _MATRIX_DENSITY_REF)
                lithology = _MATRIX_TO_LITHOLOGY[matrix]
                counts = Counter(_nearest(v, _MATRIX_DENSITY_REF) for v in gden)
                conf = counts.get(matrix, 0) / gden.size
                votes = {
                    "clastic": int(counts.get("sandstone", 0)),
                    "limestone": int(counts.get("limestone", 0)),
                    "dolomite": int(counts.get("dolomite", 0)),
                }
                if warn:
                    warnings.warn(
                        f"lithology/matrix='auto': resolved '{lithology}'/'{matrix}' via "
                        f"core_grain_density (median {median:.2f} g/cm3 of {gden.size} samples).",
                        stacklevel=2,
                    )
                return LithologyResolution(
                    lithology, matrix, "core_grain_density", int(gden.size), float(conf), votes,
                )

    porosity_curves = _BENCHMARK_CORE_POROSITY_CURVES if historical else _CORE_POROSITY_CURVES
    rhob_c, _, _ = get_curve_data_depthwise(well, ["RHOB"])
    phi_c, _, _ = get_curve_data_depthwise(well, list(porosity_curves))
    if rhob_c is None or phi_c is None:
        return None
    rhob = np.asarray(rhob_c, dtype=float)
    phi = np.asarray(phi_c, dtype=float)

    # HC-corrected pore-fluid density from a water-saturation curve, if available.
    sw_c, _, _ = get_curve_data_depthwise(well, ["SW", "SW_INPUT", "SWT-LOG", "SWA", "SWT"])
    if sw_c is not None:
        sw = np.clip(np.asarray(sw_c, dtype=float), 0.0, 1.0)
        rho_fl = np.where(np.isfinite(sw), sw * rho_fluid + (1.0 - sw) * rho_hc, rho_fluid)
    else:
        rho_fl = np.full_like(rhob, rho_fluid)

    clean = _clean_mask(well, gr_percentile)
    mask = np.isfinite(rhob) & np.isfinite(phi) & np.isfinite(rho_fl) & (phi > 0.01) & (phi < 0.40)
    if clean is not None and clean.shape == mask.shape:
        mask = mask & clean
    if int(np.count_nonzero(mask)) < _MIN_SAMPLES:
        return None
    rho_ma = (rhob[mask] - phi[mask] * rho_fl[mask]) / (1.0 - phi[mask])
    rho_ma = rho_ma[np.isfinite(rho_ma)]
    if rho_ma.size < _MIN_SAMPLES:
        return None
    labels = [_nearest(v, _MATRIX_DENSITY_REF) for v in rho_ma]
    counts = Counter(labels)
    votes = (counts.get("sandstone", 0), counts.get("limestone", 0), counts.get("dolomite", 0))
    if not historical:
        res = _finalize(*votes, "core_matrix_density", rho_ma.size, False)
        if res.confidence < _MIN_CORE_MATRIX_CONFIDENCE:
            if warn:
                warnings.warn(
                    f"lithology/matrix='auto': the core-porosity matrix density votes are split "
                    f"({res.votes}; best {res.confidence:.0%} < {_MIN_CORE_MATRIX_CONFIDENCE:.0%}), "
                    "so they do not decide the lithology.",
                    stacklevel=2,
                )
            return None
    return _finalize(*votes, "core_matrix_density", rho_ma.size, warn)


def resolve_from_minerals(well, warn: bool = True) -> Optional[LithologyResolution]:
    """Tier 1: classify from solver mineral volumes (VQUAR/VCALC/VDOLO), if present."""
    vq, _, _ = get_curve_data_depthwise(well, ["VQUAR", "VQTZ", "VSAND"])
    from petrophys._internal.benchmark import reproducing_benchmark

    # VCAL is the calcite curve MultiMineralSolver.add_to_well writes; the
    # benchmark packages were computed without it.
    calcite_names = ["VCALC", "VLIM"] if reproducing_benchmark() else ["VCALC", "VCAL", "VLIM"]
    vc, _, _ = get_curve_data_depthwise(well, calcite_names)
    vd, _, _ = get_curve_data_depthwise(well, ["VDOLO", "VDOL"])
    if vq is None or vc is None or vd is None:
        return None
    vq = np.asarray(vq, dtype=float)
    vc = np.asarray(vc, dtype=float)
    vd = np.asarray(vd, dtype=float)
    mask = np.isfinite(vq) & np.isfinite(vc) & np.isfinite(vd)
    mask &= (vq >= 0.0) & (vc >= 0.0) & (vd >= 0.0) & ((vq + vc + vd) > 0.0)
    qc = well.get("MMQC")
    if qc is not None:
        mask &= np.isfinite(qc.data) & (np.asarray(qc.data) == 0.0)
    if int(np.count_nonzero(mask)) < _MIN_SAMPLES:
        return None
    dom = np.argmax(np.vstack([vq[mask], vc[mask], vd[mask]]), axis=0)
    clastic_n = int(np.count_nonzero(dom == 0))
    carb_n = int(np.count_nonzero(dom == 1))
    dol_n = int(np.count_nonzero(dom == 2))
    return _finalize(clastic_n, carb_n, dol_n, "multi_mineral_solver",
                     int(np.count_nonzero(mask)), warn)


def resolve_well_lithology(well, warn: bool = True) -> LithologyResolution:
    """Resolve lithology/matrix: vendor log -> core matrix density -> solver -> default."""
    for resolver in (
        resolve_from_lithology_log,
        resolve_from_core_matrix,
        resolve_from_minerals,
    ):
        res = resolver(well, warn=warn)
        if res is not None:
            return res
    return _default(warn)
