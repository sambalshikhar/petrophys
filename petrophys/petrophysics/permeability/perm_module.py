"""
Main permeability estimation module.
"""

from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from petrophys.core.calibration import (
    PorosityCorrection,
    PoroPermTransform,
    core_data_from_well,
    fit_porosity_correction,
    fit_log_poro_perm_transform_from_correction,
)
from petrophys.petrophysics.permeability.timur import timur
from petrophys.petrophysics.permeability.coates_denoo import coates_denoo, wyllie_rose


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

    @property
    def n_core_points(self) -> int:
        """Return the number of matched core points used for the transform."""
        return self.perm_transform.n_points

    def predict(self, porosity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict corrected porosity and permeability for a log curve."""
        phi_corr = self.porosity_correction.predict(porosity)
        perm = np.clip(self.perm_transform.predict(phi_corr), 0.001, 100000.0)
        return phi_corr, perm


@dataclass
class PermeabilityResult:
    """Result container for permeability."""

    perm: np.ndarray
    method: str
    swi_used: Optional[Union[float, np.ndarray]] = None
    swi_source: str = "not_used"
    bvw_irr: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def estimate_apparent_swirr(
    phi: np.ndarray,
    sw: np.ndarray,
    vcl: Optional[np.ndarray] = None,
    vcl_cutoff: float = 0.35,
    phi_cutoff: float = 0.10
) -> tuple[np.ndarray, float, int]:
    """
    Estimate apparent irreducible water saturation from Buckles-style BVW.

    The apparent irreducible BVW is taken as the median of ``PHIE * SW`` in
    clean, porous intervals. Apparent ``Swirr`` is then ``BVWirr / PHIE``.
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


def resolve_swirr(
    well: "Well",
    phi: Optional[np.ndarray] = None
) -> tuple[Union[float, np.ndarray], str, Optional[float]]:
    """
    Resolve irreducible water saturation for permeability equations.

    Priority:
    1. Explicit ``SWIRR`` / ``SWI`` curves
    2. Apparent ``Swirr`` from Buckles-style BVW
    """
    if phi is None:
        phi_curve = well.get("PHIE")
        if phi_curve is None:
            raise ValueError("PHIE required to resolve Swirr")
        phi = phi_curve.data

    vcl_curve = well.get("VCL")
    vcl = vcl_curve.data if vcl_curve is not None else None

    for curve_name in ("SWIRR", "SWI", "SWIR"):
        curve = well.get(curve_name)
        if curve is None:
            continue

        data = np.asarray(curve.data, dtype=float)
        valid = np.isfinite(data) & (data > 0) & (data <= 1)
        if not np.any(valid):
            continue

        valid_fraction = np.count_nonzero(valid) / max(len(data), 1)
        if valid_fraction >= 0.5:
            filled = data.copy()
            filled[~valid] = float(np.nanmedian(data[valid]))
            return filled, curve_name.lower(), None

        return float(np.nanmedian(data[valid])), f"{curve_name.lower()}_median", None

    sw_curve = well.get("SW")
    if sw_curve is None:
        raise ValueError("SW or measured Swirr data required for permeability")

    swi_app, bvw_irr, _ = estimate_apparent_swirr(phi, sw_curve.data, vcl=vcl)
    return swi_app, "apparent_buckles", bvw_irr


def _normalize_key(value: Any) -> str:
    """Normalize a string key for matching."""
    return str(value or "").strip().lower()


def _resolve_porosity_curve_name(well: "Well") -> str:
    """Resolve the porosity curve used for permeability work."""
    for curve_name in ("PHIE", "PHIT"):
        if well.get(curve_name) is not None:
            return curve_name
    raise ValueError("PHIE or PHIT required")


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
        "n_core_points": model.n_core_points,
        "porosity_r2": model.porosity_correction.r_squared,
        "perm_r2": model.perm_transform.r_squared,
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

    try:
        core = core_data_from_well(source_well, top=top, bottom=bottom)
    except ValueError:
        return None

    if core.porosity is None or core.permeability is None:
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
    except ValueError:
        return None

    if transform.n_points < min_core_points:
        return None

    return CorePermModel(
        porosity_correction=correction,
        perm_transform=transform,
        source_well=source_well.name,
        source_field=source_well.header.field,
        source_formation=formation_name,
        source_type="same_well_core" if formation_name is None else "same_well_core_formation",
        input_curve=porosity_curve,
    )


def _model_rank(model: CorePermModel) -> tuple[float, float, float]:
    """Rank a core model by coverage first, then fit quality."""
    return (
        float(model.n_core_points),
        float(model.perm_transform.r_squared),
        float(model.porosity_correction.r_squared),
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


class PermeabilityModule:
    """Permeability estimation module."""

    def __init__(
        self,
        method: str = "auto",
        fallback_method: str = "timur",
        min_core_points: int = 3,
    ):
        self.method = method.lower()
        self.fallback_method = fallback_method.lower()
        self.min_core_points = int(min_core_points)

    def _calculate_empirical(
        self,
        well: "Well",
        phi: np.ndarray,
        swi: Optional[Union[float, np.ndarray]] = None,
        method: Optional[str] = None,
    ) -> PermeabilityResult:
        """Calculate permeability using an empirical equation."""
        empirical_method = (method or self.method).lower()

        swi_source = "user"
        bvw_irr = None
        if swi is None:
            swi, swi_source, bvw_irr = resolve_swirr(well, phi=phi)

        if empirical_method == "timur":
            perm = timur(phi, swi)
        elif empirical_method == "coates":
            perm = coates_denoo(phi, swi)
        elif empirical_method == "wyllie_rose":
            perm = wyllie_rose(phi, swi)
        else:
            raise ValueError(f"Unknown permeability method: {empirical_method}")

        return PermeabilityResult(
            perm=perm,
            method=empirical_method,
            swi_used=swi,
            swi_source=swi_source,
            bvw_irr=bvw_irr,
            metadata={"strategy": empirical_method},
        )

    def calculate(
        self,
        well: "Well",
        swi: Optional[Union[float, np.ndarray]] = None,
        phi: Optional[np.ndarray] = None,
        peer_wells: Optional[List["Well"]] = None,
        formation_tops: Optional[Any] = None,
    ) -> PermeabilityResult:
        """Calculate permeability."""
        if phi is None:
            porosity_curve = _resolve_porosity_curve_name(well)
            phi = np.asarray(well.get(porosity_curve).data, dtype=float)
        else:
            porosity_curve = _resolve_porosity_curve_name(well) if well.get("PHIE") or well.get("PHIT") else "PHIE"
            phi = np.asarray(phi, dtype=float)

        if self.method in {"timur", "coates", "wyllie_rose"}:
            return self._calculate_empirical(well, phi, swi=swi, method=self.method)

        if self.method != "auto":
            raise ValueError(f"Unknown permeability method: {self.method}")

        perm = np.full_like(phi, np.nan, dtype=float)
        phi_corecal = np.full_like(phi, np.nan, dtype=float)
        perm_core = np.full_like(phi, np.nan, dtype=float)

        same_well_phi, same_well_perm, same_well_models = _build_same_well_core_prediction(
            well,
            formation_tops=formation_tops,
            min_core_points=self.min_core_points,
            include_global=False,
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
        if np.any(unresolved) and peer_wells:
            peer_phi, peer_perm, peer_models = _build_peer_core_prediction(
                well,
                peer_wells=peer_wells,
                formation_tops=formation_tops,
                min_core_points=self.min_core_points,
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
            )
            if same_well_global is not None:
                phi_global, perm_global = same_well_global.predict(phi[unresolved])
                phi_corecal[unresolved] = phi_global
                perm_core[unresolved] = perm_global
                perm[unresolved] = perm_global
                same_well_models.append(same_well_global)
                unresolved = ~np.isfinite(perm)

        empirical_result: Optional[PermeabilityResult] = None
        if np.any(unresolved):
            empirical_result = self._calculate_empirical(
                well,
                phi,
                swi=swi,
                method=self.fallback_method,
            )
            perm[unresolved] = empirical_result.perm[unresolved]

        core_models = same_well_models + peer_models
        metadata = {
            "strategy": _describe_auto_method(
                same_well_models=same_well_models,
                peer_models=peer_models,
                used_empirical=empirical_result is not None,
                fallback_method=self.fallback_method,
            ),
            "porosity_curve": porosity_curve,
            "core_models": [_serialize_core_model(model) for model in core_models],
            "core_sections_count": int(np.count_nonzero(np.isfinite(perm_core))),
            "phi_corecal": phi_corecal if np.any(np.isfinite(phi_corecal)) else None,
            "perm_core": perm_core if np.any(np.isfinite(perm_core)) else None,
            "peer_wells_considered": int(len(peer_wells or [])),
            "same_well_core_models": int(len(same_well_models)),
            "peer_core_models": int(len(peer_models)),
            "empirical_fallback": self.fallback_method if empirical_result is not None else None,
        }

        swi_used = empirical_result.swi_used if empirical_result is not None else None
        swi_source = empirical_result.swi_source if empirical_result is not None else "not_used"
        bvw_irr = empirical_result.bvw_irr if empirical_result is not None else None

        return PermeabilityResult(
            perm=perm,
            method=metadata["strategy"],
            swi_used=swi_used,
            swi_source=swi_source,
            bvw_irr=bvw_irr,
            metadata=metadata,
        )


def calculate_permeability(
    well: "Well",
    method: str = "auto",
    **kwargs
) -> np.ndarray:
    """Calculate permeability."""
    fallback_method = kwargs.pop("fallback_method", "timur")
    min_core_points = kwargs.pop("min_core_points", 3)
    module = PermeabilityModule(
        method=method,
        fallback_method=fallback_method,
        min_core_points=min_core_points,
    )
    result = module.calculate(well, **kwargs)
    return result.perm
