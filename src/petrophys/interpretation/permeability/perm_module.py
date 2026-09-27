"""
Main permeability estimation module.
"""

from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from petrophys.interpretation.core_analysis.calibration import (
    PorosityCorrection,
    PoroPermTransform,
    core_data_from_well,
    fit_porosity_correction,
    fit_log_poro_perm_transform_from_correction,
)
from petrophys.model.constants import DefaultParameters
from petrophys.interpretation.permeability.timur import timur
from petrophys.interpretation.permeability.coates_denoo import coates_denoo, wyllie_rose
from petrophys.interpretation.permeability.lith_vcl import (
    DEFAULT_MAX_PERM_MD,
    lith_vcl_parameter_labels,
    lith_vcl_permeability,
)
from petrophys._internal.analysis_helpers import get_curve, get_curve_data_depthwise

PERMEABILITY_MAX_MD = DEFAULT_MAX_PERM_MD

SWIRR_EXPLICIT_CURVES = ("SWIRR", "SWI", "SWIR")
NMR_SWIRR_CURVES = ("SWIRR_NMR", "NMR_SWIRR", "SWIRR_CMR", "CMR_SWIRR")
NMR_TOTAL_POROSITY_CURVES = (
    "NMR_PHIT",
    "MPHI",
    "TCMR",
    "CMRPHI",
    "NMRP",
    "PHINMR",
)
NMR_BVI_CURVES = ("NMR_BVI", "MBVI", "CBVI", "CMR_BVI", "BVI")
NMR_FFI_CURVES = ("NMR_FFI", "MFFI", "CMFF", "CMR_FFI", "FFI")


def _cap_permeability(perm: np.ndarray, max_perm: float = PERMEABILITY_MAX_MD) -> np.ndarray:
    """Cap finite permeability values while preserving NaNs."""
    capped = np.asarray(perm, dtype=float).copy()
    finite = np.isfinite(capped)
    capped[finite] = np.minimum(capped[finite], max_perm)
    return capped


def _as_fraction_if_percent(values: np.ndarray) -> np.ndarray:
    """Convert percentage-like saturation values to fractions."""
    arr = np.asarray(values, dtype=float).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.5 and np.nanmax(np.abs(finite)) <= 100.0:
        arr = arr / 100.0
    return arr


def _valid_swirr(values: np.ndarray) -> np.ndarray:
    arr = _as_fraction_if_percent(values)
    return np.isfinite(arr) & (arr > 0.0) & (arr <= 1.0)


def _filled_swirr_result(
    data: np.ndarray,
    source: str,
    source_curve: Optional[str],
    metadata: Dict[str, Any],
) -> tuple[Union[float, np.ndarray], str, Optional[float], Dict[str, Any]]:
    """Return a Swirr curve when broadly populated, otherwise its valid median."""
    values = _as_fraction_if_percent(data)
    valid = _valid_swirr(values)
    valid_fraction = np.count_nonzero(valid) / max(len(values), 1)
    metadata = dict(metadata)
    metadata.update(
        {
            "source": source,
            "source_curve": source_curve,
            "valid_fraction": float(valid_fraction),
        }
    )
    if valid_fraction >= 0.5:
        filled = values.copy()
        filled[~valid] = float(np.nanmedian(values[valid]))
        return filled, source, None, metadata

    return float(np.nanmedian(values[valid])), f"{source}_median", None, metadata


@dataclass
class CorePermModel:
    """Core-derived porosity correction + poro-perm transform."""

    porosity_correction: PorosityCorrection
    perm_transform: PoroPermTransform
    source_well: str
    source_field: str
    source_formation: Optional[str] = None
    source_type: str = "same_well_core"
    input_curve: str = "PHIE"
    core_porosity_curve: Optional[str] = None
    core_permeability_curve: Optional[str] = None

    @property
    def n_core_points(self) -> int:
        """Return the number of matched core points used for the transform."""
        return self.perm_transform.n_points

    def predict(self, porosity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict corrected porosity and permeability for a log curve."""
        phi_corr = self.porosity_correction.predict(porosity)
        perm = np.clip(self.perm_transform.predict(phi_corr), 0.001, PERMEABILITY_MAX_MD)
        return phi_corr, perm


@dataclass
class PermeabilityResult:
    """Result container for permeability."""

    perm: np.ndarray
    method: str
    swi_used: Optional[Union[float, np.ndarray]] = None
    swi_source: str = "not_used"
    bvw_irr: Optional[float] = None
    swi_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def estimate_hc_swirr(
    sw: np.ndarray,
    rdep: np.ndarray,
    resflag: np.ndarray,
    rt_cutoff: float = 5.0,
    percentile: float = 5.0,
    min_samples: int = 10,
    fallback_swirr: float = 0.20,
) -> tuple[float, int, str]:
    """
    Estimate irreducible water saturation from hydrocarbon-bearing samples.

    The primary estimate is ``percentile(SW[resflag & RDEP > rt_cutoff])``.
    When the mask has too few samples, a scalar fallback Swirr is returned.
    """
    sw = np.asarray(sw, dtype=float)
    rdep = np.asarray(rdep, dtype=float)
    resflag = np.asarray(resflag, dtype=bool)
    hc_mask = resflag & np.isfinite(sw) & np.isfinite(rdep) & (rdep > rt_cutoff)
    n_samples = int(np.count_nonzero(hc_mask))
    if n_samples > min_samples:
        swirr = float(np.percentile(sw[hc_mask], percentile))
        return float(np.clip(swirr, 0.01, 1.0)), n_samples, "hc_mask_p5"
    return float(fallback_swirr), n_samples, "default_0.20"


def estimate_swirr_app(
    sw: np.ndarray,
    rdep: np.ndarray,
    resflag: np.ndarray,
    rt_cutoff: float = 5.0,
    percentile: float = 5.0,
    min_samples: int = 10,
    fallback_swirr: float = 0.20,
) -> tuple[np.ndarray, float, int, str]:
    """
    Build the SWIRR_APP curve from the hydrocarbon-mask Swirr method.

    The selected scalar Swirr is the P5 of ``SW`` where the reservoir flag is
    true and ``RDEP/RT`` exceeds the cutoff. The returned curve is a constant
    log so downstream exports can audit which Swirr value permeability used.
    """
    sw = np.asarray(sw, dtype=float)
    swirr, n_samples, source = estimate_hc_swirr(
        sw,
        rdep,
        resflag,
        rt_cutoff=rt_cutoff,
        percentile=percentile,
        min_samples=min_samples,
        fallback_swirr=fallback_swirr,
    )
    return np.full_like(sw, swirr, dtype=float), swirr, n_samples, source


def estimate_apparent_swirr(
    phi: np.ndarray,
    sw: np.ndarray,
    vcl: Optional[np.ndarray] = None,
    vcl_cutoff: float = 0.35,
    phi_cutoff: float = 0.10
) -> tuple[np.ndarray, float, int]:
    """
    Legacy Buckles-style Swirr estimate retained for direct callers.

    The dynamic workflow no longer uses this fallback; it uses
    ``estimate_hc_swirr`` through ``resolve_swirr``.
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    mask = np.isfinite(phi) & np.isfinite(sw) & (phi > phi_cutoff)
    if vcl is not None:
        vcl = np.asarray(vcl, dtype=float)
        mask &= np.isfinite(vcl) & (vcl <= vcl_cutoff)
    if np.count_nonzero(mask) < 10:
        mask = np.isfinite(phi) & np.isfinite(sw) & (phi > max(phi_cutoff * 0.7, 0.05))
    if np.count_nonzero(mask) == 0:
        raise ValueError("Cannot estimate apparent Swirr without valid PHIE/SW samples")
    bvw_irr = float(np.nanmedian(phi[mask] * sw[mask]))
    phi_safe = np.where(phi <= 1e-4, 1e-4, phi)
    swi_app = np.clip(bvw_irr / phi_safe, 0.01, 1.0)
    return swi_app, bvw_irr, int(np.count_nonzero(mask))


def _resolve_resflag(well: "Well", phi: np.ndarray) -> tuple[np.ndarray, str]:
    """Resolve a reservoir flag, deriving a fallback only for direct callers."""
    resflag, resflag_source, _ = get_curve_data_depthwise(well, ["RESFLAG", "NET_RES"])
    if resflag is not None:
        return np.asarray(resflag, dtype=float) > 0.5, resflag_source

    vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
    if vcl is not None:
        resflag = (
            np.isfinite(vcl)
            & np.isfinite(phi)
            & (vcl < DefaultParameters.VCL_CUTOFF_CARBONATE)  # strict clean-sample mask (0.30)
            & (phi > DefaultParameters.PHIE_CUTOFF_CLASTIC)
        )
        return resflag, "derived_vcl_phie"

    resflag = np.isfinite(phi) & (phi > DefaultParameters.PHIE_CUTOFF_CLASTIC)
    return resflag, "derived_phie"


def _resolve_nmr_swirr_details(
    well: "Well",
) -> Optional[tuple[Union[float, np.ndarray], str, Optional[float], Dict[str, Any]]]:
    """Resolve Swirr from NMR curves without creating a separate output curve."""
    data, source_curve, meta = get_curve_data_depthwise(well, NMR_SWIRR_CURVES)
    if data is not None:
        valid = _valid_swirr(data)
        if np.any(valid):
            return _filled_swirr_result(
                data,
                "nmr_swirr",
                source_curve,
                {
                    "method": "nmr_swirr_curve",
                    "input_curves": [source_curve] if source_curve else [],
                    "source_counts": meta.get("source_counts", {}),
                },
            )

    total, total_source, total_meta = get_curve_data_depthwise(
        well, NMR_TOTAL_POROSITY_CURVES
    )
    if total is None:
        return None

    bvi, bvi_source, bvi_meta = get_curve_data_depthwise(well, NMR_BVI_CURVES)
    ffi, ffi_source, ffi_meta = get_curve_data_depthwise(well, NMR_FFI_CURVES)

    method = None
    if bvi is not None:
        numerator = bvi
        method = "nmr_bvi_over_total_porosity"
    elif ffi is not None:
        numerator = total - ffi
        method = "nmr_total_minus_ffi_over_total_porosity"
    else:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        swirr = np.where(
            np.isfinite(total) & (total > 0) & np.isfinite(numerator),
            numerator / total,
            np.nan,
        )
    swirr = np.where(swirr > 0.0, swirr, np.nan)
    valid = _valid_swirr(swirr)
    if not np.any(valid):
        return None

    input_curves = [name for name in (total_source, bvi_source, ffi_source) if name]
    return _filled_swirr_result(
        swirr,
        "nmr_derived",
        total_source,
        {
            "method": method,
            "input_curves": input_curves,
            "nmr_total_porosity_curve": total_source,
            "nmr_total_porosity_source_counts": total_meta.get("source_counts", {}),
            "nmr_bvi_curve": bvi_source,
            "nmr_bvi_source_counts": bvi_meta.get("source_counts", {}),
            "nmr_ffi_curve": ffi_source,
            "nmr_ffi_source_counts": ffi_meta.get("source_counts", {}),
        },
    )


def _resolve_swirr_details(
    well: "Well",
    phi: Optional[np.ndarray] = None,
) -> tuple[Union[float, np.ndarray], str, Optional[float], Dict[str, Any]]:
    """Resolve Swirr plus trace metadata."""
    if phi is None:
        phi, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        if phi is None:
            raise ValueError("PHIE or PHIT with finite samples required to resolve Swirr")
    phi = np.asarray(phi, dtype=float)

    data, swirr_source_curve, swirr_meta = get_curve_data_depthwise(
        well, SWIRR_EXPLICIT_CURVES
    )
    if data is not None:
        valid = _valid_swirr(data)
        if np.any(valid):
            return _filled_swirr_result(
                data,
                "swirr",
                swirr_source_curve,
                {"source_counts": swirr_meta.get("source_counts", {})},
            )

    nmr_result = _resolve_nmr_swirr_details(well)
    if nmr_result is not None:
        return nmr_result

    sw, sw_source, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
    rdep, rdep_source, rdep_meta = get_curve_data_depthwise(well, ["RDEP", "RT"])
    if sw is None or rdep is None:
        return 0.20, "default_0.20", None, {
            "reason": "SW or RDEP/RT unavailable",
            "fallback_swirr": 0.20,
            "swirr_app": np.full_like(phi, 0.20, dtype=float),
            "swirr_app_method": "default_0.20",
        }

    resflag, resflag_source = _resolve_resflag(well, phi)
    swirr_app, swirr, n_samples, source = estimate_swirr_app(
        sw,
        rdep,
        resflag,
    )
    return swirr, source, None, {
        "resflag_source": resflag_source,
        "sw_curve": sw_source,
        "rt_curve": rdep_source,
        "rt_source_counts": rdep_meta.get("source_counts", {}),
        "rt_cutoff": 5.0,
        "percentile": 5.0,
        "min_samples": 10,
        "hc_mask_samples": n_samples,
        "fallback_swirr": 0.20,
        "swirr_app": swirr_app,
        "swirr_app_method": source,
    }


def resolve_swirr(
    well: "Well",
    phi: Optional[np.ndarray] = None
) -> tuple[Union[float, np.ndarray], str, Optional[float]]:
    """
    Resolve irreducible water saturation for permeability equations.

    Priority:
    1. Explicit ``SWIRR`` / ``SWI`` curves
    2. NMR-derived Swirr from NMR curves when available
    3. ``P5(SW)`` where reservoir flag is true and ``RDEP > 5 ohm.m``
    4. Scalar fallback ``0.20``
    """
    swirr, source, bvw_irr, _ = _resolve_swirr_details(well, phi=phi)
    return swirr, source, bvw_irr


def _normalize_key(value: Any) -> str:
    """Normalize a string key for matching."""
    return str(value or "").strip().lower()


#: Permeability methods accepted by :class:`PermeabilityModule`.
PERMEABILITY_METHODS = ("auto", "timur", "coates", "wyllie_rose", "lith_vcl")

#: Accepted alternative spellings and their canonical permeability method.
PERMEABILITY_METHOD_ALIASES = {
    "coates_denoo": "coates",
    "coates-denoo": "coates",
    "wyllie-rose": "wyllie_rose",
    "lithology_vcl": "lith_vcl",
    "lith-vcl": "lith_vcl",
    "lith_vcl_perm": "lith_vcl",
    "dunite": "lith_vcl",
    "dunite_basic": "lith_vcl",
}


def _normalize_perm_method(value: str) -> str:
    """Normalize permeability method aliases."""
    method = str(value or "").strip().lower()
    return PERMEABILITY_METHOD_ALIASES.get(method, method)


def _resolve_porosity_curve_name(well: "Well") -> str:
    """Resolve the porosity curve used for permeability work."""
    for curve_name in ("PHIE", "PHIT"):
        if get_curve(well, [curve_name]) is not None:
            return curve_name
    raise ValueError("PHIE or PHIT with finite samples required")


def _resolve_lithology_input(
    well: "Well",
    lithology: Optional[Union[str, np.ndarray]] = None,
) -> tuple[Optional[Union[str, np.ndarray]], str]:
    """Resolve the best available lithology input curve for permeability."""
    if lithology is not None:
        return lithology, "provided_lithology"

    lith_curve = get_curve(well, ["LITHOLOGY_GEOLINK"])
    if lith_curve is not None:
        try:
            return np.asarray(lith_curve.data, dtype=float), lith_curve.mnemonic
        except (TypeError, ValueError):
            return np.asarray(lith_curve.data, dtype=object), lith_curve.mnemonic

    lith_class_curve = well.get("LITHO_CLASS")
    if lith_class_curve is not None:
        lith_class = np.asarray(lith_class_curve.data, dtype=object)
        if any(str(item).strip() for item in lith_class):
            return lith_class, lith_class_curve.mnemonic

    computed_lith_curve = get_curve(well, ["LITHO"])
    if computed_lith_curve is not None:
        lith_codes = np.asarray(computed_lith_curve.data, dtype=float)
        lith = np.full(lith_codes.shape, "sandstone", dtype=object)
        lith[np.isclose(lith_codes, 2)] = "limestone"
        lith[np.isclose(lith_codes, 3)] = "dolomite"
        lith[np.isclose(lith_codes, 5)] = "anhydrite"
        lith[np.isclose(lith_codes, 7)] = "coal"
        return lith, computed_lith_curve.mnemonic

    return None, "not_available"


def _resolve_lith_vcl_inputs(
    well: "Well",
    phi: np.ndarray,
    lithology: Optional[Union[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[Union[str, np.ndarray]], Dict[str, Any]]:
    """Resolve PHIT, VCL, and lithology inputs for lith_vcl permeability."""
    phit, phit_source, _ = get_curve_data_depthwise(well, ["PHIT"])
    if phit is not None:
        phit = np.asarray(phit, dtype=float)
    else:
        phit = np.asarray(phi, dtype=float)
        phit_source = "PHIT_missing_using_permeability_porosity"

    vcl, vcl_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
    if vcl is None:
        raise ValueError("VCL or VCLAY with finite samples required for lith_vcl permeability")
    vcl = np.asarray(vcl, dtype=float)

    lith, lith_source = _resolve_lithology_input(well, lithology=lithology)
    if lith_source == "not_available":
        lith_source = "default_sandstone"

    labels = lith_vcl_parameter_labels(lith)
    unique_labels = sorted({str(label) for label in np.ravel(labels) if str(label)})
    metadata = {
        "phit_curve": phit_source,
        "vcl_curve": vcl_source,
        "lithology_curve": lith_source,
        "lithology_parameter_groups": unique_labels,
        "permvclscalar": 1.2,
        "equation": "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)",
    }
    return phit, vcl, lith, metadata


def _get_formation_tops(
    well: "Well",
    formation_tops: Optional[Any] = None,
) -> Optional[Any]:
    """Get formation tops from an explicit argument or well parameters."""
    if formation_tops is not None:
        return formation_tops
    return well.get_parameter("formation_tops")


def _serialize_core_model(model: CorePermModel) -> Dict[str, Any]:
    """Convert a core-derived permeability model to simple metadata."""
    return {
        "source_well": model.source_well,
        "source_field": model.source_field,
        "source_formation": model.source_formation,
        "source_type": model.source_type,
        "input_curve": model.input_curve,
        "core_porosity_curve": model.core_porosity_curve,
        "core_permeability_curve": model.core_permeability_curve,
        "n_core_points": model.n_core_points,
        "porosity_correction": {
            "A": model.porosity_correction.A,
            "B": model.porosity_correction.B,
        },
        "perm_transform": {
            "A": model.perm_transform.A,
            "B": model.perm_transform.B,
        },
    }


def _fit_core_perm_model(
    source_well: "Well",
    formation_tops: Optional[Any] = None,
    formation_name: Optional[str] = None,
    min_core_points: int = 3,
    rejected_models: Optional[List[Dict[str, Any]]] = None,
) -> Optional[CorePermModel]:
    """
    Fit a core-derived porosity correction and poro-perm transform.

    If ``formation_name`` is provided, the fit is limited to that formation
    interval using the supplied ``formation_tops``.
    """
    porosity_curve = _resolve_porosity_curve_name(source_well)
    top = None
    bottom = None

    if formation_name is not None:
        tops = _get_formation_tops(source_well, formation_tops)
        if tops is None:
            return None

        interval = tops.get_by_name(formation_name)
        if interval is None:
            return None

        top = interval.top_depth
        bottom = interval.bottom_depth

    def _reject(reason: str, **details: Any) -> None:
        if rejected_models is None:
            return
        rejected_models.append(
            {
                "source_well": source_well.name,
                "source_field": source_well.header.field,
                "source_formation": formation_name,
                "reason": reason,
                **details,
            }
        )

    try:
        core = core_data_from_well(source_well, top=top, bottom=bottom)
    except ValueError:
        _reject("no_core_curves")
        return None

    if core.porosity is None or core.permeability is None:
        _reject(
            "missing_core_porosity_or_permeability",
            core_porosity_curve=core.metadata.get("core_porosity_curve"),
            core_permeability_curve=core.metadata.get("core_permeability_curve"),
        )
        return None

    try:
        correction = fit_porosity_correction(
            source_well,
            core,
            log_curve=porosity_curve,
        )
        transform = fit_log_poro_perm_transform_from_correction(
            source_well,
            core,
            correction,
            porosity_curve=porosity_curve,
        )
    except ValueError as exc:
        _reject(
            "fit_failed",
            error=str(exc),
            core_porosity_curve=core.metadata.get("core_porosity_curve"),
            core_permeability_curve=core.metadata.get("core_permeability_curve"),
        )
        return None

    if transform.n_points < min_core_points:
        _reject(
            "too_few_core_points",
            n_core_points=transform.n_points,
            min_core_points=min_core_points,
            core_porosity_curve=core.metadata.get("core_porosity_curve"),
            core_permeability_curve=core.metadata.get("core_permeability_curve"),
        )
        return None

    return CorePermModel(
        porosity_correction=correction,
        perm_transform=transform,
        source_well=source_well.name,
        source_field=source_well.header.field,
        source_formation=formation_name,
        source_type="same_well_core" if formation_name is None else "same_well_core_formation",
        input_curve=porosity_curve,
        core_porosity_curve=core.metadata.get("core_porosity_curve"),
        core_permeability_curve=core.metadata.get("core_permeability_curve"),
    )


def _model_rank(model: CorePermModel) -> tuple[float, float, float]:
    """Rank a core model by matched core coverage."""
    return (
        float(model.n_core_points),
        0.0,
        0.0,
    )


def _apply_core_model(
    phi_input: np.ndarray,
    mask: np.ndarray,
    model: CorePermModel,
    phi_corecal: np.ndarray,
    perm_core: np.ndarray,
):
    """Apply a fitted core model into masked portions of full-length arrays."""
    if not np.any(mask):
        return
    phi_corr, perm = model.predict(phi_input[mask])
    phi_corecal[mask] = phi_corr
    perm_core[mask] = perm


def _build_same_well_core_prediction(
    well: "Well",
    formation_tops: Optional[Any] = None,
    min_core_points: int = 3,
    include_global: bool = True,
    rejected_models: Optional[List[Dict[str, Any]]] = None,
) -> tuple[np.ndarray, np.ndarray, List[CorePermModel]]:
    """Build same-well core-calibrated porosity and permeability."""
    porosity_curve = _resolve_porosity_curve_name(well)
    phi_input = np.asarray(well.get(porosity_curve).data, dtype=float)
    depth = np.asarray(well.depth, dtype=float)

    phi_corecal = np.full_like(phi_input, np.nan, dtype=float)
    perm_core = np.full_like(phi_input, np.nan, dtype=float)
    models: List[CorePermModel] = []

    tops = _get_formation_tops(well, formation_tops)
    if tops is not None:
        for interval in tops.intervals:
            model = _fit_core_perm_model(
                well,
                formation_tops=tops,
                formation_name=interval.name,
                min_core_points=min_core_points,
                rejected_models=rejected_models,
            )
            if model is None:
                continue

            mask = interval.contains(depth)
            _apply_core_model(phi_input, mask, model, phi_corecal, perm_core)
            models.append(model)

    unresolved = ~np.isfinite(perm_core)
    if include_global and np.any(unresolved):
        model = _fit_core_perm_model(
            well,
            formation_tops=None,
            formation_name=None,
            min_core_points=min_core_points,
            rejected_models=rejected_models,
        )
        if model is not None:
            _apply_core_model(phi_input, unresolved, model, phi_corecal, perm_core)
            models.append(model)

    return phi_corecal, perm_core, models


def _build_peer_core_prediction(
    well: "Well",
    peer_wells: Optional[List["Well"]] = None,
    formation_tops: Optional[Any] = None,
    min_core_points: int = 3,
    rejected_models: Optional[List[Dict[str, Any]]] = None,
) -> tuple[np.ndarray, np.ndarray, List[CorePermModel]]:
    """
    Build permeability from peer wells in the same field and formation.

    This requires ``FormationTops`` on the target well and on candidate peer
    wells (stored on each well as ``well.set_parameter("formation_tops", tops)``).
    """
    porosity_curve = _resolve_porosity_curve_name(well)
    phi_input = np.asarray(well.get(porosity_curve).data, dtype=float)
    depth = np.asarray(well.depth, dtype=float)
    phi_corecal = np.full_like(phi_input, np.nan, dtype=float)
    perm_core = np.full_like(phi_input, np.nan, dtype=float)
    models: List[CorePermModel] = []

    if not peer_wells:
        return phi_corecal, perm_core, models

    target_field = _normalize_key(well.header.field)
    target_tops = _get_formation_tops(well, formation_tops)
    if not target_field or target_tops is None:
        return phi_corecal, perm_core, models

    for interval in target_tops.intervals:
        best_model: Optional[CorePermModel] = None

        for peer_well in peer_wells:
            if peer_well is None or peer_well is well:
                continue

            if _normalize_key(peer_well.header.field) != target_field:
                continue

            peer_tops = _get_formation_tops(peer_well)
            if peer_tops is None or peer_tops.get_by_name(interval.name) is None:
                continue

            model = _fit_core_perm_model(
                peer_well,
                formation_tops=peer_tops,
                formation_name=interval.name,
                min_core_points=min_core_points,
                rejected_models=rejected_models,
            )
            if model is None:
                continue

            model.source_type = "peer_field_formation"
            if best_model is None or _model_rank(model) > _model_rank(best_model):
                best_model = model

        if best_model is None:
            continue

        mask = interval.contains(depth)
        _apply_core_model(phi_input, mask, best_model, phi_corecal, perm_core)
        models.append(best_model)

    return phi_corecal, perm_core, models


def _describe_auto_method(
    same_well_models: List[CorePermModel],
    peer_models: List[CorePermModel],
    used_empirical: bool,
    fallback_method: str,
) -> str:
    """Build a readable method label for the final permeability curve."""
    parts: List[str] = []
    if same_well_models:
        parts.append("same_well_core")
    if peer_models:
        parts.append("peer_field_formation_core")
    if used_empirical or not parts:
        parts.append(fallback_method)
    return "+".join(parts)


def _auto_empirical_methods(fallback_method: str) -> List[str]:
    """Return empirical methods in workflow priority order."""
    fallback = _normalize_perm_method(fallback_method)
    methods: List[str] = ["coates"]
    if fallback not in {"", "coates", "timur", "lith_vcl"}:
        methods.append(fallback)
    methods.append("timur")
    methods.append("lith_vcl")
    return list(dict.fromkeys(methods))


class PermeabilityModule:
    """Permeability estimation module."""

    def __init__(
        self,
        method: str = "auto",
        fallback_method: str = "coates",
        min_core_points: int = 3,
        use_peer_core: bool = False,
    ):
        self.method = _normalize_perm_method(method)
        self.fallback_method = _normalize_perm_method(fallback_method)
        self.min_core_points = int(min_core_points)
        self.use_peer_core = bool(use_peer_core)

    def _calculate_empirical(
        self,
        well: "Well",
        phi: np.ndarray,
        swi: Optional[Union[float, np.ndarray]] = None,
        method: Optional[str] = None,
        lithology: Optional[Union[str, np.ndarray]] = None,
    ) -> PermeabilityResult:
        """Calculate permeability using an empirical equation."""
        empirical_method = _normalize_perm_method(method or self.method)

        swi_source = "user" if swi is not None else "not_required"
        bvw_irr = None
        swi_metadata: Dict[str, Any] = {"source": "user"} if swi is not None else {"source": "not_required"}

        if empirical_method == "timur":
            if swi is None:
                swi, swi_source, bvw_irr, swi_metadata = _resolve_swirr_details(well, phi=phi)
            perm = timur(phi, swi)
        elif empirical_method == "coates":
            if swi is None:
                swi, swi_source, bvw_irr, swi_metadata = _resolve_swirr_details(well, phi=phi)
            perm = coates_denoo(phi, swi)
        elif empirical_method == "wyllie_rose":
            if swi is None:
                swi, swi_source, bvw_irr, swi_metadata = _resolve_swirr_details(well, phi=phi)
            perm = wyllie_rose(phi, swi)
        elif empirical_method == "lith_vcl":
            phit, vcl, lith, lith_metadata = _resolve_lith_vcl_inputs(well, phi, lithology=lithology)
            perm = lith_vcl_permeability(phit, vcl, lith)
            swi_metadata = {"source": "not_required", "reason": "lith_vcl permeability does not use Swirr"}
        else:
            raise ValueError(f"Unknown permeability method: {empirical_method}")

        perm = _cap_permeability(perm)
        finite_count = int(np.count_nonzero(np.isfinite(perm)))
        if finite_count == 0:
            raise ValueError(f"{empirical_method} permeability produced no finite samples")

        _, phi_source, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        input_curves = [phi_source or "PHIE"]
        method_metadata: Dict[str, Any] = {}
        if empirical_method == "lith_vcl":
            method_metadata = lith_metadata
            input_curves = [
                lith_metadata["phit_curve"],
                lith_metadata["vcl_curve"],
                lith_metadata["lithology_curve"],
            ]
        elif empirical_method == "coates":
            method_metadata = {
                "equation": "10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2",
                "coefficient": 10000.0,
                "porosity_exponent": 4.0,
            }
        elif swi_source != "user":
            input_curves.extend(
                swi_metadata.get("input_curves")
                or [swi_metadata.get("source_curve") or "SWIRR/SW"]
            )

        if empirical_method == "coates" and swi_source != "user":
            input_curves.extend(
                swi_metadata.get("input_curves")
                or [swi_metadata.get("source_curve") or "SWIRR/SW"]
            )

        return PermeabilityResult(
            perm=perm,
            method=empirical_method,
            swi_used=swi,
            swi_source=swi_source,
            bvw_irr=bvw_irr,
            swi_metadata=swi_metadata,
            metadata={
                "strategy": empirical_method,
                "empirical_perm": perm,
                "empirical_method": empirical_method,
                "empirical_attempted_methods": [empirical_method],
                "empirical_finite_count": finite_count,
                "input_curves": input_curves,
                "method_metadata": method_metadata,
                "swi_metadata": swi_metadata,
                "perm_max_clip_md": PERMEABILITY_MAX_MD,
            },
        )

    def calculate_core_correction(
        self,
        well: "Well",
        phi: Optional[np.ndarray] = None,
        peer_wells: Optional[List["Well"]] = None,
        formation_tops: Optional[Any] = None,
    ) -> PermeabilityResult:
        """Calculate same-well/peer core-corrected PHIE_CC and PERM_CC only."""
        if phi is None:
            phi, porosity_curve, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
            if phi is None:
                raise ValueError("PHIE or PHIT with finite samples required")
            phi = np.asarray(phi, dtype=float)
        else:
            porosity_curve = _resolve_porosity_curve_name(well) if get_curve(well, ["PHIE", "PHIT"]) is not None else "PHIE"
            phi = np.asarray(phi, dtype=float)

        phi_corecal = np.full_like(phi, np.nan, dtype=float)
        perm_core = np.full_like(phi, np.nan, dtype=float)
        core_rejected_models: List[Dict[str, Any]] = []

        same_well_phi, same_well_perm, same_well_models = _build_same_well_core_prediction(
            well,
            formation_tops=formation_tops,
            min_core_points=self.min_core_points,
            include_global=False,
            rejected_models=core_rejected_models,
        )
        same_mask = np.isfinite(same_well_perm)
        if np.any(same_mask):
            phi_corecal[same_mask] = same_well_phi[same_mask]
            perm_core[same_mask] = same_well_perm[same_mask]

        unresolved = ~np.isfinite(perm_core)
        peer_models: List[CorePermModel] = []
        if np.any(unresolved) and peer_wells and self.use_peer_core:
            peer_phi, peer_perm, peer_models = _build_peer_core_prediction(
                well,
                peer_wells=peer_wells,
                formation_tops=formation_tops,
                min_core_points=self.min_core_points,
                rejected_models=core_rejected_models,
            )
            peer_mask = unresolved & np.isfinite(peer_perm)
            if np.any(peer_mask):
                phi_corecal[peer_mask] = peer_phi[peer_mask]
                perm_core[peer_mask] = peer_perm[peer_mask]
                unresolved = ~np.isfinite(perm_core)

        if np.any(unresolved):
            same_well_global = _fit_core_perm_model(
                well,
                formation_tops=None,
                formation_name=None,
                min_core_points=self.min_core_points,
                rejected_models=core_rejected_models,
            )
            if same_well_global is not None:
                phi_global, perm_global = same_well_global.predict(phi[unresolved])
                phi_corecal[unresolved] = phi_global
                perm_core[unresolved] = perm_global
                same_well_models.append(same_well_global)

        perm_core = _cap_permeability(perm_core)
        core_models = same_well_models + peer_models
        metadata = {
            "strategy": _describe_auto_method(
                same_well_models=same_well_models,
                peer_models=peer_models,
                used_empirical=False,
                fallback_method=self.fallback_method,
            ),
            "porosity_curve": porosity_curve,
            "core_models": [_serialize_core_model(model) for model in core_models],
            "core_sections_count": int(np.count_nonzero(np.isfinite(perm_core))),
            "phi_corecal": phi_corecal if np.any(np.isfinite(phi_corecal)) else None,
            "perm_core": perm_core if np.any(np.isfinite(perm_core)) else None,
            "peer_wells_considered": int(len(peer_wells or [])),
            "peer_core_enabled": bool(self.use_peer_core),
            "same_well_core_models": int(len(same_well_models)),
            "peer_core_models": int(len(peer_models)),
            "core_min_points": int(self.min_core_points),
            "core_rejected_models": core_rejected_models,
            "perm_max_clip_md": PERMEABILITY_MAX_MD,
        }
        return PermeabilityResult(
            perm=perm_core,
            method=metadata["strategy"],
            swi_source="not_used",
            metadata=metadata,
        )

    def calculate(
        self,
        well: "Well",
        swi: Optional[Union[float, np.ndarray]] = None,
        phi: Optional[np.ndarray] = None,
        lithology: Optional[Union[str, np.ndarray]] = None,
        peer_wells: Optional[List["Well"]] = None,
        formation_tops: Optional[Any] = None,
    ) -> PermeabilityResult:
        """Calculate permeability."""
        if phi is None:
            phi, porosity_curve, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
            if phi is None:
                raise ValueError("PHIE or PHIT with finite samples required")
            phi = np.asarray(phi, dtype=float)
        else:
            porosity_curve = _resolve_porosity_curve_name(well) if get_curve(well, ["PHIE", "PHIT"]) is not None else "PHIE"
            phi = np.asarray(phi, dtype=float)

        if self.method in {"timur", "coates", "wyllie_rose", "lith_vcl"}:
            return self._calculate_empirical(well, phi, swi=swi, method=self.method, lithology=lithology)

        if self.method != "auto":
            raise ValueError(f"Unknown permeability method: {self.method}")

        perm = np.full_like(phi, np.nan, dtype=float)
        phi_corecal = np.full_like(phi, np.nan, dtype=float)
        perm_core = np.full_like(phi, np.nan, dtype=float)
        empirical_result: Optional[PermeabilityResult] = None
        empirical_error: Optional[str] = None
        core_rejected_models: List[Dict[str, Any]] = []

        empirical_methods = _auto_empirical_methods(self.fallback_method)

        empirical_errors: List[str] = []
        empirical_attempted_methods: List[str] = []
        for empirical_method in empirical_methods:
            empirical_attempted_methods.append(empirical_method)
            try:
                empirical_result = self._calculate_empirical(
                    well,
                    phi,
                    swi=swi,
                    method=empirical_method,
                    lithology=lithology,
                )
                break
            except ValueError as exc:
                empirical_errors.append(f"{empirical_method}: {exc}")
        if empirical_errors:
            empirical_error = " | ".join(empirical_errors)

        same_well_phi, same_well_perm, same_well_models = _build_same_well_core_prediction(
            well,
            formation_tops=formation_tops,
            min_core_points=self.min_core_points,
            include_global=False,
            rejected_models=core_rejected_models,
        )
        same_mask = np.isfinite(same_well_perm)
        if np.any(same_mask):
            perm[same_mask] = same_well_perm[same_mask]
            phi_corecal[same_mask] = same_well_phi[same_mask]
            perm_core[same_mask] = same_well_perm[same_mask]

        unresolved = ~np.isfinite(perm)
        peer_phi = np.full_like(phi, np.nan, dtype=float)
        peer_perm = np.full_like(phi, np.nan, dtype=float)
        peer_models: List[CorePermModel] = []
        if np.any(unresolved) and peer_wells and self.use_peer_core:
            peer_phi, peer_perm, peer_models = _build_peer_core_prediction(
                well,
                peer_wells=peer_wells,
                formation_tops=formation_tops,
                min_core_points=self.min_core_points,
                rejected_models=core_rejected_models,
            )
            peer_mask = unresolved & np.isfinite(peer_perm)
            if np.any(peer_mask):
                perm[peer_mask] = peer_perm[peer_mask]
                phi_corecal[peer_mask] = peer_phi[peer_mask]
                perm_core[peer_mask] = peer_perm[peer_mask]
                unresolved = ~np.isfinite(perm)

        if np.any(unresolved):
            same_well_global = _fit_core_perm_model(
                well,
                formation_tops=None,
                formation_name=None,
                min_core_points=self.min_core_points,
                rejected_models=core_rejected_models,
            )
            if same_well_global is not None:
                phi_global, perm_global = same_well_global.predict(phi[unresolved])
                phi_corecal[unresolved] = phi_global
                perm_core[unresolved] = perm_global
                perm[unresolved] = perm_global
                same_well_models.append(same_well_global)
                unresolved = ~np.isfinite(perm)

        if np.any(unresolved):
            if empirical_result is None:
                raise ValueError(empirical_error or "SW or measured Swirr data required for permeability")
            perm[unresolved] = empirical_result.perm[unresolved]

        perm = _cap_permeability(perm)
        perm_core = _cap_permeability(perm_core)
        core_models = same_well_models + peer_models
        metadata = {
            "strategy": _describe_auto_method(
                same_well_models=same_well_models,
                peer_models=peer_models,
                used_empirical=empirical_result is not None,
                fallback_method=empirical_result.method if empirical_result is not None else self.fallback_method,
            ),
            "porosity_curve": porosity_curve,
            "core_models": [_serialize_core_model(model) for model in core_models],
            "core_sections_count": int(np.count_nonzero(np.isfinite(perm_core))),
            "phi_corecal": phi_corecal if np.any(np.isfinite(phi_corecal)) else None,
            "perm_core": perm_core if np.any(np.isfinite(perm_core)) else None,
            "peer_wells_considered": int(len(peer_wells or [])),
            "peer_core_enabled": bool(self.use_peer_core),
            "same_well_core_models": int(len(same_well_models)),
            "peer_core_models": int(len(peer_models)),
            "core_min_points": int(self.min_core_points),
            "core_rejected_models": core_rejected_models,
            "empirical_fallback": empirical_result.method if empirical_result is not None else None,
            "empirical_perm": empirical_result.perm if empirical_result is not None else None,
            "empirical_method": empirical_result.method if empirical_result is not None else self.fallback_method,
            "empirical_error": empirical_error,
            "empirical_attempted_methods": empirical_attempted_methods,
            "swi_metadata": empirical_result.swi_metadata if empirical_result is not None else {},
            "input_curves": empirical_result.metadata.get("input_curves", []) if empirical_result is not None else [],
            "method_metadata": empirical_result.metadata.get("method_metadata", {}) if empirical_result is not None else {},
            "perm_max_clip_md": PERMEABILITY_MAX_MD,
        }

        swi_used = empirical_result.swi_used if empirical_result is not None else None
        swi_source = empirical_result.swi_source if empirical_result is not None else "not_used"
        bvw_irr = empirical_result.bvw_irr if empirical_result is not None else None
        swi_metadata = empirical_result.swi_metadata if empirical_result is not None else {}

        return PermeabilityResult(
            perm=perm,
            method=metadata["strategy"],
            swi_used=swi_used,
            swi_source=swi_source,
            bvw_irr=bvw_irr,
            swi_metadata=swi_metadata,
            metadata=metadata,
        )


def calculate_permeability(
    well: "Well",
    method: str = "auto",
    **kwargs
) -> np.ndarray:
    """Calculate permeability."""
    fallback_method = kwargs.pop("fallback_method", "coates")
    min_core_points = kwargs.pop("min_core_points", 3)
    use_peer_core = kwargs.pop("use_peer_core", False)
    module = PermeabilityModule(
        method=method,
        fallback_method=fallback_method,
        min_core_points=min_core_points,
        use_peer_core=use_peer_core,
    )
    result = module.calculate(well, **kwargs)
    return result.perm
