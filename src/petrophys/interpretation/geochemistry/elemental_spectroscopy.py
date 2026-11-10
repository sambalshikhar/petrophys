"""Interpret processed multi-vendor borehole elemental spectroscopy.

The supported inputs are processed elemental dry-weight fractions and relative
elemental yields.  Raw gamma-ray spectra are deliberately out of scope because
their inversion requires tool response matrices, energy calibration, and
environmental corrections.

Default mnemonic families are drawn from the Energistics PWLS curve catalog:

* SLB ``DW*``/``SL*`` dry-weight products and capture/inelastic yields.
* Baker Hughes ``WF*`` dry-weight products.
* Halliburton GEM corrected and uncorrected elemental yields.

Description-based matching provides a vendor-neutral fallback for Weatherford
and other deliveries whose public product literature defines the measurement
but not a stable LAS mnemonic.

References
----------
Hertzog, R., Colson, L., Seeman, B., et al. (1989). Geochemical logging with
spectrometry tools. *SPE Formation Evaluation* 4(2), 153-162.

Grau, J.A., Schweitzer, J.S., Ellis, D.V. and Hertzog, R.C. (1989). A
geological model for gamma-ray spectroscopy logging measurements. *Nuclear
Geophysics* 3(4), 351-359.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from petrophys._internal.analysis_helpers import base_curve_mnemonic
from petrophys.interpretation.geochemistry.xrf import XRFData
from petrophys.model.data_types import DataType

ELEMENT_NAMES: Mapping[str, str] = {
    "AL": "aluminum",
    "BA": "barium",
    "C": "carbon",
    "CA": "calcium",
    "CL": "chlorine",
    "FE": "iron",
    "GD": "gadolinium",
    "H": "hydrogen",
    "K": "potassium",
    "MG": "magnesium",
    "MN": "manganese",
    "NA": "sodium",
    "NI": "nickel",
    "NI_CR": "nickel-chromium",
    "O": "oxygen",
    "S": "sulfur",
    "SI": "silicon",
    "TI": "titanium",
}


# Ordered from more explicit/processed products toward generic fallbacks.
_DRY_WEIGHT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "AL": ("ELEM_AL", "WFAL", "WFALP2", "DWAL", "SLAL", "AL_WT"),
    "BA": ("ELEM_BA", "WFBA", "DWBA", "SLBA", "BA_WT"),
    "C": ("ELEM_C", "WFCC", "WFC", "DWCO", "DWC", "SLCO", "C_WT"),
    "CA": ("ELEM_CA", "WFCA", "DWCA", "SLCA", "CA_WT"),
    "CL": ("ELEM_CL", "WFCL", "DWCL", "SLCL", "CL_WT"),
    "FE": ("ELEM_FE", "WFFE", "DWFE", "SLFE", "FE_WT"),
    "GD": ("ELEM_GD", "WFGD", "DWGD", "SLGD", "GD_WT"),
    "H": ("ELEM_H", "WFH", "DWHY", "DWH", "SLHY", "H_WT"),
    "K": ("ELEM_K", "WFK", "DWK", "SLK", "K_WT"),
    "MG": ("ELEM_MG", "WFMG", "DWMG", "SLMG", "MG_WT"),
    "MN": ("ELEM_MN", "WFMN", "DWMN", "SLMN", "MN_WT"),
    "NA": ("ELEM_NA", "WFNA", "DWNA", "SLNA", "NA_WT"),
    "O": ("ELEM_O", "WFO", "DWO", "SLO", "O_WT"),
    "S": ("ELEM_S", "WFS", "DWSU", "SLSU", "S_WT"),
    "SI": ("ELEM_SI", "WFSI", "DWSI", "SLSI", "SI_WT"),
    "TI": ("ELEM_TI", "WFTI", "DWTI", "SLTI", "TI_WT"),
}


_CAPTURE_YIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "CA": ("CCAC", "CCA"),
    "CL": ("CCHLC", "CCHL", "CDCL"),
    "FE": ("CFEC", "CFE"),
    "GD": ("CGDC", "CGD"),
    "H": ("CHYC", "CHY", "CH"),
    "K": ("CKC", "CK"),
    "NI_CR": ("CNCC", "CNC"),
    "BA": ("CBAC", "CBA"),
    "S": ("CSUC", "CSUL", "CSU"),
    "SI": ("CSIC", "CSI"),
    "TI": ("CTIC", "CTI"),
}


_INELASTIC_YIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "AL": ("IALC", "IAL"),
    "C": ("ICC", "IC"),
    "CA": ("ICAC", "ICA"),
    "FE": ("IFEC", "IFE"),
    "MG": ("IMGC", "IMG"),
    "O": ("IOC", "IO"),
    "S": ("ISUC", "ISU"),
    "SI": ("ISIC", "ISI"),
}


# Halliburton GEM channels whose short names are too ambiguous to accept unless
# their descriptions explicitly identify the GEM elemental measurement.
_GEM_DESCRIPTION_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "AL": ("CALC", "CAL"),
    "BA": ("CBAC", "CBA"),
    "MG": ("CMGC", "CMG"),
    "MN": ("CMNC", "CMN"),
    "NA": ("CNAC", "CNA"),
}

_UNCERTAINTY_WORDS = re.compile(
    r"\b(?:uncertaint(?:y|ies)|error|upper\s+bound|lower\s+bound|maximum|minimum)\b",
    re.I,
)
_BOREHOLE_WORDS = re.compile(r"\bborehole\b", re.I)


@dataclass
class ElementalSpectroscopyResult:
    """Resolved elemental products and interpretation diagnostics."""

    dry_weight_fractions: Dict[str, np.ndarray] = field(default_factory=dict)
    relative_yields: Dict[str, np.ndarray] = field(default_factory=dict)
    uncertainties: Dict[str, np.ndarray] = field(default_factory=dict)
    derived: Dict[str, np.ndarray] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    conversions: Dict[str, str] = field(default_factory=dict)
    output_curves: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def input_curves(self) -> list[str]:
        """Unique source mnemonics consumed by the interpretation."""
        return sorted(set(self.sources.values()))


@dataclass
class OxideClosureResult:
    """Element and oxide fractions produced by calibrated oxide closure."""

    element_fractions: Dict[str, np.ndarray]
    oxide_fractions: Dict[str, np.ndarray]
    scale_factor: np.ndarray
    closure: np.ndarray


#: Display labels of the fraction ratios, whose names hold three elements.
_RATIO_LABELS = {"Ca_Ca_Mg": "Ca/(Ca+Mg)", "Al_Al_Si": "Al/(Al+Si)"}

def _description(curve: Any) -> str:
    metadata = getattr(curve, "metadata", None)
    return str(getattr(metadata, "description", "") or "").strip()


def _unit_key(unit: str) -> str:
    text = str(unit or "").strip().lower()
    text = text.replace("weight", "wt").replace("percent", "%")
    return re.sub(r"[\s_]+", "", text)


def _family_matches(mnemonic: str, alias: str) -> bool:
    return mnemonic == alias or mnemonic.startswith(f"{alias}_")


def _element_in_description(element: str, description: str) -> bool:
    if element == "NI_CR":
        return bool(re.search(r"\bnickel[-\s]+chromium\b", description, re.I))
    if element == "NI" and re.search(r"\bchromium\b", description, re.I):
        return False
    name = ELEMENT_NAMES.get(element, element.lower())
    return bool(re.search(rf"\b{re.escape(name)}\b", description, re.I))


def _is_dry_weight_description(element: str, description: str) -> bool:
    if not _element_in_description(element, description):
        return False
    return bool(
        re.search(
            r"\b(?:dry[-\s]+weight|weight\s+fraction|elemental\s+concentration)\b",
            description,
            re.I,
        )
    )


def _yield_mode_from_description(element: str, description: str) -> Optional[str]:
    if not _element_in_description(element, description):
        return None
    if re.search(
        r"\binelastic\b.*\b(?:yield|element)|\b(?:yield|element)\b.*\binelastic\b",
        description,
        re.I,
    ):
        return "INELASTIC"
    if re.search(
        r"\bcapture\b.*\b(?:yield|element)|\b(?:yield|element)\b.*\bcapture\b",
        description,
        re.I,
    ):
        return "CAPTURE"
    if re.search(r"\bGEM\b", description, re.I):
        return "GEM"
    return None


def _finite_count(curve: Any) -> int:
    try:
        values = np.asarray(curve.data, dtype=float)
    except (TypeError, ValueError):
        return 0
    return int(np.count_nonzero(np.isfinite(values)))


def _processing_variant(mnemonic: str) -> str:
    """Return a processing-family token used to pair values and uncertainty."""
    base = base_curve_mnemonic(mnemonic)
    walk = re.search(r"(?:^|_)(WALK\d+)(?:_|$)", base)
    if walk:
        return walk.group(1)
    if re.search(r"(?:^|_)ACQ(?:_|$)", base):
        return "ACQ"
    if re.search(r"(?:^|_)20(?:_|$)", base):
        return "20"
    return ""


def _dry_weight_score(
    element: str,
    curve_name: str,
    curve: Any,
    aliases: Sequence[str],
) -> Optional[float]:
    base = base_curve_mnemonic(curve_name)
    description = _description(curve)
    alias_index = next(
        (index for index, alias in enumerate(aliases) if _family_matches(base, alias)),
        None,
    )
    description_match = _is_dry_weight_description(element, description)
    if alias_index is None and not description_match:
        return None
    if _UNCERTAINTY_WORDS.search(description) or _BOREHOLE_WORDS.search(description):
        return None
    if re.search(r"\b(?:relative\s+yield|count\s+rate|spectrum)\b", description, re.I):
        return None

    finite_count = _finite_count(curve)
    if finite_count == 0:
        return None

    score = 0.0
    if alias_index is not None:
        score += 200.0 - float(alias_index)
    if description_match:
        score += 70.0
    if re.search(r"\bnormalized\b", description, re.I):
        score += 8.0
    if re.search(r"WALK2", base, re.I) and not re.search(r"ACQ|RT", base, re.I):
        score += 6.0
    if re.search(r"ACQ|REAL.?TIME|_RT\b", base + " " + description, re.I):
        score -= 2.0
    score += min(np.log10(finite_count + 1.0), 6.0)
    return score


def _yield_score(
    element: str,
    mode: str,
    curve_name: str,
    curve: Any,
    aliases: Sequence[str],
    *,
    require_gem_description: bool = False,
) -> Optional[float]:
    base = base_curve_mnemonic(curve_name)
    description = _description(curve)
    alias_index = next(
        (index for index, alias in enumerate(aliases) if _family_matches(base, alias)),
        None,
    )
    description_mode = _yield_mode_from_description(element, description)
    if require_gem_description and description_mode != "GEM":
        return None
    if alias_index is None and description_mode != mode:
        return None
    if _UNCERTAINTY_WORDS.search(description) or _BOREHOLE_WORDS.search(description):
        return None
    if _is_dry_weight_description(element, description):
        return None

    finite_count = _finite_count(curve)
    if finite_count == 0:
        return None

    score = 0.0
    if alias_index is not None:
        score += 180.0 - float(alias_index)
    if description_mode == mode:
        score += 70.0
    elif description_mode == "GEM":
        score += 50.0
    if re.search(r"\bcorrected\b", description, re.I) or base.endswith("C"):
        score += 8.0
    if re.search(r"WALK2", base, re.I) and not re.search(r"ACQ|RT", base, re.I):
        score += 5.0
    score += min(np.log10(finite_count + 1.0), 6.0)
    return score


def _resolve_best(
    well: Any,
    score_fn: Any,
) -> Optional[Tuple[str, Any]]:
    best: Optional[Tuple[float, str, Any]] = None
    for curve_name in getattr(well, "curves", []):
        curve = well.get(curve_name)
        if curve is None:
            continue
        score = score_fn(curve_name, curve)
        if score is None:
            continue
        candidate = (score, str(curve_name), curve)
        if (
            best is None
            or candidate[0] > best[0]
            or (candidate[0] == best[0] and candidate[1] < best[1])
        ):
            best = candidate
    if best is None:
        return None
    return best[1], best[2]


def _convert_weight_fraction(
    source_name: str,
    curve: Any,
) -> Tuple[np.ndarray, str, Optional[str]]:
    values = np.asarray(curve.data, dtype=float).copy()
    unit = str(getattr(curve, "unit", "") or "")
    key = _unit_key(unit)
    if "%" in key:
        return values / 100.0, "percent_to_fraction", None
    if "ppm" in key or "ug/g" in key or "mg/kg" in key:
        return values / 1_000_000.0, "ppm_to_fraction", None
    if key in {
        "fraction",
        "mass/mass",
        "wt/wt",
        "g/g",
        "kg/kg",
        "kgf/kgf",
        "dimensionless",
        "unitless",
        "v/v",
    }:
        return values, "identity_fraction", None

    finite = values[np.isfinite(values)]
    conversion = "assumed_fraction"
    if finite.size and float(np.nanpercentile(np.abs(finite), 95)) > 1.5:
        values = values / 100.0
        conversion = "inferred_percent_to_fraction"
    warning = (
        f"{source_name}: elemental dry-weight unit {unit!r} is ambiguous; "
        f"{conversion.replace('_', ' ')} was applied"
    )
    return values, conversion, warning


def _add_output_curve(
    well: Any,
    name: str,
    values: np.ndarray,
    unit: str,
    description: str,
    result: ElementalSpectroscopyResult,
    *,
    source_name: str = "",
    conversion: str = "",
) -> None:
    if well.get(name) is not None:
        return
    curve = well.add_curve(
        name,
        values,
        unit=unit,
        description=description,
        data_type=DataType.COMPUTED,
    )
    curve.metadata.source = "elemental_spectroscopy"
    curve.metadata.custom.update(
        {
            "source_mnemonic": source_name,
            "conversion": conversion,
            "method": "processed_elemental_spectroscopy",
        }
    )
    result.output_curves.append(name)


def _aliases_with_overrides(
    defaults: Mapping[str, Tuple[str, ...]],
    overrides: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Tuple[str, ...]]:
    output: Dict[str, Tuple[str, ...]] = {}
    overrides = overrides or {}
    unknown = set(overrides) - set(ELEMENT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown elemental symbols in alias overrides: {sorted(unknown)}"
        )
    for element in ELEMENT_NAMES:
        output[element] = tuple(
            dict.fromkeys(
                [
                    *(
                        str(alias).strip().upper()
                        for alias in overrides.get(element, ())
                    ),
                    *defaults.get(element, ()),
                ]
            )
        )
    return output


def has_elemental_spectroscopy_inputs(well: Any) -> bool:
    """Return whether processed elemental concentrations or yields are present."""
    dry_aliases = _aliases_with_overrides(_DRY_WEIGHT_ALIASES, None)
    capture_aliases = _aliases_with_overrides(_CAPTURE_YIELD_ALIASES, None)
    inelastic_aliases = _aliases_with_overrides(_INELASTIC_YIELD_ALIASES, None)

    for element, aliases in dry_aliases.items():
        if (
            _resolve_best(
                well,
                lambda name, curve, e=element, a=aliases: _dry_weight_score(
                    e, name, curve, a
                ),
            )
            is not None
        ):
            return True
    for mode, defaults in (
        ("CAPTURE", capture_aliases),
        ("INELASTIC", inelastic_aliases),
    ):
        for element, aliases in defaults.items():
            if (
                _resolve_best(
                    well,
                    lambda name, curve, e=element, m=mode, a=aliases: _yield_score(
                        e, m, name, curve, a
                    ),
                )
                is not None
            ):
                return True
    for element, aliases in _GEM_DESCRIPTION_ALIASES.items():
        if (
            _resolve_best(
                well,
                lambda name, curve, e=element, a=aliases: _yield_score(
                    e,
                    "GEM",
                    name,
                    curve,
                    a,
                    require_gem_description=True,
                ),
            )
            is not None
        ):
            return True
    return False


def oxide_closure(
    relative_yields: Mapping[str, np.ndarray],
    sensitivities: Mapping[str, float],
    oxide_factors: Mapping[str, float],
) -> OxideClosureResult:
    """Convert calibrated relative yields to elemental dry-weight fractions.

    The calculation follows ``W_i = F * Y_i / S_i`` with
    ``F = 1 / sum(X_i * Y_i / S_i)``.  ``S_i`` is the tool sensitivity and
    ``X_i`` is the oxide-association factor.  Both must be supplied by the
    caller; no vendor calibration is guessed.  The closure needs every
    supplied yield, so a depth where any yield is missing is NaN in all
    outputs rather than closed over the remaining elements.
    """
    keys = [str(key).strip().upper() for key in relative_yields]
    if not keys:
        raise ValueError("relative_yields must not be empty")
    if set(keys) - {str(key).strip().upper() for key in sensitivities}:
        missing = sorted(
            set(keys) - {str(key).strip().upper() for key in sensitivities}
        )
        raise ValueError(f"Missing sensitivities for: {missing}")
    if set(keys) - {str(key).strip().upper() for key in oxide_factors}:
        missing = sorted(
            set(keys) - {str(key).strip().upper() for key in oxide_factors}
        )
        raise ValueError(f"Missing oxide factors for: {missing}")

    sensitivity_map = {
        str(key).strip().upper(): float(value) for key, value in sensitivities.items()
    }
    oxide_map = {
        str(key).strip().upper(): float(value) for key, value in oxide_factors.items()
    }
    arrays = {
        str(key).strip().upper(): np.asarray(value, dtype=float)
        for key, value in relative_yields.items()
    }
    shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))

    adjusted: Dict[str, np.ndarray] = {}
    denominator = np.zeros(shape, dtype=float)
    finite_all = np.ones(shape, dtype=bool)
    for key, values in arrays.items():
        sensitivity = sensitivity_map[key]
        factor = oxide_map[key]
        if not np.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError(f"Sensitivity for {key} must be positive and finite")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Oxide factor for {key} must be positive and finite")
        broadcast = np.broadcast_to(values, shape)
        component = broadcast / sensitivity
        adjusted[key] = component
        finite = np.isfinite(component)
        denominator[finite] += factor * component[finite]
        finite_all &= finite

    scale = np.full(shape, np.nan)
    valid = finite_all & np.isfinite(denominator) & (denominator > 0)
    scale[valid] = 1.0 / denominator[valid]

    element_fractions: Dict[str, np.ndarray] = {}
    oxide_fractions: Dict[str, np.ndarray] = {}
    closure = np.zeros(shape, dtype=float)
    closure[~valid] = np.nan
    for key, component in adjusted.items():
        element = component * scale
        oxide = oxide_map[key] * element
        element_fractions[key] = element
        oxide_fractions[key] = oxide
        closure[valid] += np.where(np.isfinite(oxide[valid]), oxide[valid], 0.0)

    return OxideClosureResult(
        element_fractions=element_fractions,
        oxide_fractions=oxide_fractions,
        scale_factor=scale,
        closure=closure,
    )


def interpret_elemental_spectroscopy(
    well: Any,
    *,
    add_curves: bool = True,
    dry_weight_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    capture_yield_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    inelastic_yield_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    mass_balance_tolerance: float = 0.05,
) -> ElementalSpectroscopyResult:
    """Resolve processed elemental products and derive screening diagnostics."""
    if not np.isfinite(mass_balance_tolerance) or mass_balance_tolerance < 0:
        raise ValueError("mass_balance_tolerance must be finite and non-negative")

    dry_aliases = _aliases_with_overrides(_DRY_WEIGHT_ALIASES, dry_weight_aliases)
    capture_aliases = _aliases_with_overrides(
        _CAPTURE_YIELD_ALIASES,
        capture_yield_aliases,
    )
    inelastic_aliases = _aliases_with_overrides(
        _INELASTIC_YIELD_ALIASES,
        inelastic_yield_aliases,
    )
    result = ElementalSpectroscopyResult()

    for element, aliases in dry_aliases.items():
        resolved = _resolve_best(
            well,
            lambda name, curve, e=element, a=aliases: _dry_weight_score(
                e, name, curve, a
            ),
        )
        if resolved is None:
            continue
        source_name, curve = resolved
        values, conversion, warning = _convert_weight_fraction(source_name, curve)
        result.dry_weight_fractions[element] = values
        output_name = f"ELEM_{element}"
        result.units[output_name] = "wt_frac"
        result.sources[output_name] = source_name
        result.conversions[output_name] = conversion
        if warning:
            result.warnings.append(warning)
        if add_curves:
            _add_output_curve(
                well,
                output_name,
                values,
                "wt_frac",
                f"Standardized dry-weight fraction of {ELEMENT_NAMES[element]} from {source_name}",
                result,
                source_name=source_name,
                conversion=conversion,
            )

    for mode, alias_map in (
        ("CAPTURE", capture_aliases),
        ("INELASTIC", inelastic_aliases),
    ):
        for element, aliases in alias_map.items():
            resolved = _resolve_best(
                well,
                lambda name, curve, e=element, m=mode, a=aliases: _yield_score(
                    e, m, name, curve, a
                ),
            )
            if resolved is None:
                continue
            source_name, curve = resolved
            values = np.asarray(curve.data, dtype=float).copy()
            key = f"{mode}_{element}"
            output_name = f"ELEM_{key}"
            result.relative_yields[key] = values
            result.units[output_name] = "relative_yield"
            result.sources[output_name] = source_name
            result.conversions[output_name] = "identity_relative_yield"
            if add_curves:
                _add_output_curve(
                    well,
                    output_name,
                    values,
                    "relative_yield",
                    f"Standardized {mode.lower()} relative yield of {ELEMENT_NAMES.get(element, element)} from {source_name}",
                    result,
                    source_name=source_name,
                    conversion="identity_relative_yield",
                )

    # Resolve Halliburton GEM aliases that are unsafe without their descriptions.
    for element, aliases in _GEM_DESCRIPTION_ALIASES.items():
        if any(key.endswith(f"_{element}") for key in result.relative_yields):
            continue
        resolved = _resolve_best(
            well,
            lambda name, curve, e=element, a=aliases: _yield_score(
                e,
                "GEM",
                name,
                curve,
                a,
                require_gem_description=True,
            ),
        )
        if resolved is None:
            continue
        source_name, curve = resolved
        values = np.asarray(curve.data, dtype=float).copy()
        key = f"GEM_{element}"
        output_name = f"ELEM_{key}"
        result.relative_yields[key] = values
        result.units[output_name] = "relative_yield"
        result.sources[output_name] = source_name
        result.conversions[output_name] = "identity_relative_yield"
        if add_curves:
            _add_output_curve(
                well,
                output_name,
                values,
                "relative_yield",
                f"Standardized GEM relative yield of {ELEMENT_NAMES[element]} from {source_name}",
                result,
                source_name=source_name,
                conversion="identity_relative_yield",
            )

    # Statistical uncertainties for processed dry-weight products.  This is
    # intentionally description-driven so lower/upper-bound curves are not
    # mislabeled as one-sigma uncertainty.
    for element, values in result.dry_weight_fractions.items():
        dry_source = result.sources.get(f"ELEM_{element}", "")
        dry_variant = _processing_variant(dry_source)

        def uncertainty_score(
            name: str, curve: Any, e: str = element
        ) -> Optional[float]:
            description = _description(curve)
            if not (
                _element_in_description(e, description)
                and re.search(
                    r"\bstatistical\s+uncertainty\b|\bone[-\s]+sigma\b",
                    description,
                    re.I,
                )
            ):
                return None
            finite_count = _finite_count(curve)
            if not finite_count:
                return None
            score = 150.0 + min(np.log10(finite_count + 1.0), 6.0)
            candidate_variant = _processing_variant(name)
            if dry_variant and candidate_variant == dry_variant:
                score += 30.0
            elif dry_variant and candidate_variant:
                score -= 10.0
            return score

        resolved_uncertainty = _resolve_best(
            well,
            uncertainty_score,
        )
        if resolved_uncertainty is None:
            continue
        source_name, curve = resolved_uncertainty
        uncertainty, conversion, warning = _convert_weight_fraction(source_name, curve)
        result.uncertainties[element] = uncertainty
        output_name = f"ELEM_{element}_SIGMA"
        result.units[output_name] = "wt_frac"
        result.sources[output_name] = source_name
        result.conversions[output_name] = conversion
        if warning:
            result.warnings.append(warning)
        if add_curves:
            _add_output_curve(
                well,
                output_name,
                uncertainty,
                "wt_frac",
                f"Statistical uncertainty of elemental {ELEMENT_NAMES[element]} dry-weight fraction",
                result,
                source_name=source_name,
                conversion=conversion,
            )

    def add_derived(name: str, values: np.ndarray, unit: str, description: str) -> None:
        result.derived[name] = values
        result.units[name] = unit
        if add_curves:
            _add_output_curve(
                well,
                name,
                values,
                unit,
                description,
                result,
                conversion="derived",
            )

    if result.dry_weight_fractions:
        stack = np.vstack(list(result.dry_weight_fractions.values()))
        finite_count = np.sum(np.isfinite(stack), axis=0)
        measured_sum = np.nansum(stack, axis=0)
        measured_sum[finite_count == 0] = np.nan
        add_derived(
            "ELEM_MEASURED_SUM",
            measured_sum,
            "wt_frac",
            "Sum of available processed elemental dry-weight fractions; may be incomplete",
        )

        range_flag = np.full(measured_sum.shape, np.nan)
        valid_rows = finite_count > 0
        out_of_range = np.any(
            np.isfinite(stack)
            & ((stack < 0.0) | (stack > 1.0 + mass_balance_tolerance)),
            axis=0,
        )
        range_flag[valid_rows] = out_of_range[valid_rows].astype(float)
        add_derived(
            "ELEM_RANGE_QC_FLAG",
            range_flag,
            "flag",
            "Elemental dry-weight value outside physical tolerance",
        )

        excess_flag = np.full(measured_sum.shape, np.nan)
        excess_flag[valid_rows] = (
            measured_sum[valid_rows] > 1.0 + mass_balance_tolerance
        ).astype(float)
        add_derived(
            "ELEM_MASS_BALANCE_EXCESS_FLAG",
            excess_flag,
            "flag",
            "Measured elemental dry-weight sum exceeds unity tolerance",
        )

        well_depth = getattr(well, "depth", None)
        if well_depth is None:
            depth = np.arange(stack.shape[1], dtype=float)
        else:
            depth = np.asarray(well_depth, dtype=float)
        symbols = {
            element.capitalize(): values
            for element, values in result.dry_weight_fractions.items()
        }
        elemental = XRFData(
            depth=depth, elements=symbols, units={symbol: "wt_frac" for symbol in symbols},
        )
        for ratio_name, values in elemental.elemental_ratios().items():
            if not np.any(np.isfinite(values)):
                continue
            output_name = f"ELEM_RATIO_{ratio_name.upper()}"
            add_derived(
                output_name,
                values,
                "ratio",
                f"Elemental spectroscopy ratio {_RATIO_LABELS.get(ratio_name, ratio_name.replace('_', '/'))}",
            )
        for proxy_name, values in elemental.mineral_proxies().items():
            if not np.any(np.isfinite(values)):
                continue
            output_name = f"ELEM_{proxy_name.upper()}"
            add_derived(
                output_name,
                values,
                "index",
                f"Elemental spectroscopy screening proxy: {proxy_name.replace('_', ' ')}",
            )

    return result


__all__ = [
    "ELEMENT_NAMES",
    "ElementalSpectroscopyResult",
    "OxideClosureResult",
    "has_elemental_spectroscopy_inputs",
    "interpret_elemental_spectroscopy",
    "oxide_closure",
]
