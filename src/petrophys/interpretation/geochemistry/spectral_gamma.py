"""Vendor-neutral interpretation of processed spectral gamma-ray logs.

The module consumes processed potassium, thorium, uranium, total spectral
gamma-ray, and uranium-free gamma-ray curves.  It does not invert raw detector
spectra; that operation requires tool-specific calibration and response data.

Mnemonic families are based on the Energistics Practical Well Log Standard
(PWLS) curve catalog and public service-company curve dictionaries.  A name is
never the only evidence for ambiguous one-letter curves: units and curve
descriptions are also checked before a candidate is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from petrophys._internal.analysis_helpers import base_curve_mnemonic
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.data_types import DataType

# Common API sensitivity coefficients for concentration logs.  Potassium is in
# percent; thorium and uranium are in ppm.  They remain configurable because a
# service-company calibrated API curve should be preferred when supplied.
DEFAULT_API_COEFFICIENTS: Mapping[str, float] = {
    "POTASSIUM": 16.32,
    "THORIUM": 3.93,
    "URANIUM": 8.09,
}


_ALIASES: Mapping[str, Tuple[str, ...]] = {
    # SLB: POTA/HFK, THOR, URAN.  Baker Hughes: K/K40/KC, TH/TH232/THC,
    # U/U238/UC.  Halliburton commonly delivers POTA/THOR/URAN.
    "POTASSIUM": ("POTASSIUM", "POTA", "KC", "K40", "HFK", "K"),
    "THORIUM": ("THORIUM", "THOR", "THC", "TH232", "TH"),
    "URANIUM": ("URANIUM", "URAN", "UC", "U238", "U"),
    # GRSL/GRSLC and KTH/KTHC are present in Baker Hughes PWLS mappings;
    # GRTO is present in Halliburton mappings.  KTH candidates require API-unit
    # or explicit gamma-minus-uranium description evidence.
    "SGR": ("SGR", "HSGR", "SGRC", "GRSL", "GRSLC", "GRTO", "TOTA"),
    "CGR": ("CGR", "HCGR", "GRKT", "KTHC", "KTH"),
}

_GENERIC_ALIASES = {"K", "TH", "U", "KTH", "KTHC", "TOTA"}

_DESCRIPTION_PATTERNS: Mapping[str, Tuple[re.Pattern[str], ...]] = {
    "POTASSIUM": (
        re.compile(r"\bformation\s+potassium\b", re.I),
        re.compile(r"\bpotassium\s+(?:content|concentration)\b", re.I),
    ),
    "THORIUM": (
        re.compile(r"\bformation\s+thorium\b", re.I),
        re.compile(r"\bthorium\s+(?:content|concentration)\b", re.I),
    ),
    "URANIUM": (
        re.compile(r"\bformation\s+uranium\b", re.I),
        re.compile(r"\buranium\s+(?:content|concentration)\b", re.I),
    ),
    "SGR": (
        re.compile(r"\bspectr(?:al|oscopy)\s+gamma\s+ray\b", re.I),
        re.compile(r"\bgamma\s+ray\s+(?:api\s+)?total\b", re.I),
        re.compile(r"\bgamma\s+ray\s+from\s+(?:the\s+)?spectrum\b", re.I),
    ),
    "CGR": (
        re.compile(r"\bgamma\s+ray\s+minus\s+uranium\b", re.I),
        re.compile(r"\bstripped\s+potassium[-\s]+thorium\b", re.I),
        re.compile(
            r"\bgamma\s+ray\s+contribution\s+from\s+thorium\s+and\s+potassium\b", re.I
        ),
        re.compile(r"\bcomputed\s+gamma\s+ray\b", re.I),
    ),
}

_REJECT_DESCRIPTION = re.compile(
    r"\b(?:error|uncertaint(?:y|ies)|upper\s+limit|lower\s+limit|count\s+rate|"
    r"dry\s+weight|relative\s+yield|statistical\s+uncertainty)\b",
    re.I,
)
_BOREHOLE_POTASSIUM = re.compile(r"\bborehole\s+potassium\b", re.I)


@dataclass
class SpectralGammaResult:
    """Processed spectral-gamma measurements and derived diagnostics."""

    measurements: Dict[str, np.ndarray] = field(default_factory=dict)
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


def _description(curve: Any) -> str:
    metadata = getattr(curve, "metadata", None)
    return str(getattr(metadata, "description", "") or "").strip()


def _unit_key(unit: str) -> str:
    text = str(unit or "").strip().lower()
    text = text.replace("weight", "wt").replace("percent", "%")
    return re.sub(r"[\s_]+", "", text)


def _is_api_unit(unit: str) -> bool:
    key = _unit_key(unit)
    return key in {"api", "gapi", "apiu", "apiunits", "apigammaray"}


def _is_concentration_unit(property_name: str, unit: str) -> bool:
    key = _unit_key(unit)
    if not key:
        return False
    if property_name == "POTASSIUM":
        return (
            "%" in key
            or "ppm" in key
            or key
            in {
                "fraction",
                "mass/mass",
                "wt/wt",
                "g/g",
                "kg/kg",
                "kgf/kgf",
                "dimensionless",
                "unitless",
            }
        )
    return (
        "ppm" in key
        or "ug/g" in key
        or "mg/kg" in key
        or "%" in key
        or key
        in {
            "fraction",
            "mass/mass",
            "wt/wt",
            "g/g",
            "kg/kg",
            "kgf/kgf",
            "dimensionless",
            "unitless",
        }
    )


def _matches_description(property_name: str, description: str) -> bool:
    return any(
        pattern.search(description) for pattern in _DESCRIPTION_PATTERNS[property_name]
    )


def _candidate_score(
    property_name: str,
    curve_name: str,
    curve: Any,
    aliases: Sequence[str],
) -> Optional[float]:
    base = base_curve_mnemonic(curve_name)
    description = _description(curve)
    unit = str(getattr(curve, "unit", "") or "")
    description_match = _matches_description(property_name, description)
    alias_match = base in aliases
    if not alias_match and not description_match:
        return None
    if _REJECT_DESCRIPTION.search(description):
        return None
    if property_name == "POTASSIUM" and _BOREHOLE_POTASSIUM.search(description):
        return None

    # Do not scan large array channels unless their mnemonic or description is
    # actually relevant.  A native spectroscopy delivery can contain thousands
    # of flattened detector-bin curves.
    values = np.asarray(getattr(curve, "data", []), dtype=float)
    finite_count = int(np.count_nonzero(np.isfinite(values)))
    if finite_count == 0:
        return None

    if property_name in {"SGR", "CGR"}:
        unit_match = _is_api_unit(unit)
        # KTH/KTHC may be either a dimensionless ratio or an API gamma-minus-U
        # curve.  Require unambiguous unit/description evidence.
        if base in {"KTH", "KTHC", "TOTA"} and not (
            unit_match or re.search(r"gamma\s+ray|minus\s+uranium", description, re.I)
        ):
            return None
    else:
        unit_match = _is_concentration_unit(property_name, unit)
        if base in _GENERIC_ALIASES and not (unit_match or description_match):
            return None

    score = 0.0
    if alias_match:
        score += 200.0 - float(aliases.index(base))
    if description_match:
        score += 60.0
    if unit_match:
        score += 30.0
    if re.search(r"\b(?:corrected|formation)\b", description, re.I):
        score += 8.0
    if base.endswith("C") and base not in {"CGR"}:
        score += 3.0
    score += min(np.log10(finite_count + 1.0), 6.0)
    return score


def _resolve_curve(
    well: Any,
    property_name: str,
    extra_aliases: Iterable[str] = (),
) -> Optional[Tuple[str, Any]]:
    aliases = tuple(
        dict.fromkeys(
            [
                *(alias.upper() for alias in extra_aliases),
                *_ALIASES[property_name],
            ]
        )
    )
    best: Optional[Tuple[float, str, Any]] = None
    for curve_name in getattr(well, "curves", []):
        curve = well.get(curve_name)
        if curve is None:
            continue
        score = _candidate_score(property_name, curve_name, curve, aliases)
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


def _convert_concentration(
    property_name: str,
    source_name: str,
    curve: Any,
) -> Tuple[np.ndarray, str, str, Optional[str]]:
    values = np.asarray(curve.data, dtype=float).copy()
    unit = str(getattr(curve, "unit", "") or "")
    key = _unit_key(unit)
    historical = reproducing_benchmark()
    fraction_keys = {"fraction", "mass/mass", "wt/wt", "g/g", "kg/kg", "kgf/kgf"}
    if not historical:
        fraction_keys |= {"v/v", "frac", "dec", "decimal"}
    # A unit the checks below do not recognise is assumed, so it is reported; the
    # benchmark packages warned only on blank or dimensionless units.
    ambiguous = not historical or not key or key in {"dimensionless", "unitless"}

    if property_name == "POTASSIUM":
        target_unit = "%"
        if "ppm" in key or "ug/g" in key or "mg/kg" in key:
            return values / 10_000.0, target_unit, "ppm_to_percent", None
        if "%" in key:
            return values, target_unit, "identity_percent", None
        if key in fraction_keys:
            return values * 100.0, target_unit, "fraction_to_percent", None
        warning = None
        if ambiguous:
            warning = (
                f"{source_name}: potassium unit {unit!r} is ambiguous; "
                "industry spectral-gamma convention (%) was assumed"
            )
        return values, target_unit, "assumed_percent", warning

    target_unit = "ppm"
    if "ppm" in key or "ug/g" in key or "mg/kg" in key:
        return values, target_unit, "identity_ppm", None
    if "%" in key:
        return values * 10_000.0, target_unit, "percent_to_ppm", None
    if key in fraction_keys:
        return values * 1_000_000.0, target_unit, "fraction_to_ppm", None
    warning = None
    if ambiguous:
        warning = (
            f"{source_name}: {property_name.lower()} unit {unit!r} is ambiguous; "
            "industry spectral-gamma convention (ppm) was assumed"
        )
    return values, target_unit, "assumed_ppm", warning


def _convert_api(
    source_name: str,
    curve: Any,
) -> Tuple[np.ndarray, str, str, Optional[str]]:
    values = np.asarray(curve.data, dtype=float).copy()
    unit = str(getattr(curve, "unit", "") or "")
    warning = None
    if not _is_api_unit(unit):
        warning = f"{source_name}: gamma-ray unit {unit!r} was assumed to be API"
    return values, "API", "identity_api", warning


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            numerator,
            denominator,
            out=output,
            where=np.isfinite(denominator) & (denominator != 0),
        )
    output[~np.isfinite(output)] = np.nan
    return output


def _add_output_curve(
    well: Any,
    name: str,
    values: np.ndarray,
    unit: str,
    description: str,
    result: SpectralGammaResult,
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
    curve.metadata.source = "spectral_gamma"
    curve.metadata.custom.update(
        {
            "source_mnemonic": source_name,
            "conversion": conversion,
            "method": "processed_spectral_gamma",
        }
    )
    result.output_curves.append(name)


def has_spectral_gamma_inputs(
    well: Any,
    alias_overrides: Optional[Mapping[str, Sequence[str]]] = None,
) -> bool:
    """Return whether a well contains a defensible processed spectral input."""
    overrides = alias_overrides or {}
    return any(
        _resolve_curve(well, name, overrides.get(name, ())) is not None
        for name in _ALIASES
    )


def interpret_spectral_gamma(
    well: Any,
    *,
    add_curves: bool = True,
    alias_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    api_coefficients: Optional[Mapping[str, float]] = None,
    uranium_fraction_threshold: float = 0.5,
) -> SpectralGammaResult:
    """Interpret processed K-Th-U and spectral gamma-ray curves.

    Parameters
    ----------
    well
        :class:`petrophys.model.Well`-like object.
    add_curves
        Add standardized and derived curves to ``well``.
    alias_overrides
        Optional canonical-name to additional-mnemonics mapping.  Supported
        canonical names are ``POTASSIUM``, ``THORIUM``, ``URANIUM``, ``SGR``,
        and ``CGR``.
    api_coefficients
        K, Th, and U sensitivities used to reconstruct API gamma ray.  Defaults
        to 16.32 API/% K, 3.93 API/ppm Th, and 8.09 API/ppm U.
    uranium_fraction_threshold
        Threshold for ``SGR_URANIUM_EXCESS_FLAG``.
    """
    if not 0.0 <= float(uranium_fraction_threshold) <= 1.0:
        raise ValueError("uranium_fraction_threshold must be between 0 and 1")

    overrides = alias_overrides or {}
    unknown = set(overrides) - set(_ALIASES)
    if unknown:
        raise ValueError(f"Unknown spectral-gamma canonical names: {sorted(unknown)}")

    result = SpectralGammaResult()
    for property_name in _ALIASES:
        resolved = _resolve_curve(well, property_name, overrides.get(property_name, ()))
        if resolved is None:
            continue
        source_name, curve = resolved
        if property_name in {"SGR", "CGR"}:
            values, unit, conversion, warning = _convert_api(source_name, curve)
        else:
            values, unit, conversion, warning = _convert_concentration(
                property_name,
                source_name,
                curve,
            )
        result.measurements[property_name] = values
        result.units[property_name] = unit
        result.sources[property_name] = source_name
        result.conversions[property_name] = conversion
        if warning:
            result.warnings.append(warning)
        if add_curves:
            _add_output_curve(
                well,
                property_name,
                values,
                unit,
                f"Standardized processed spectral-gamma {property_name.lower()} from {source_name}",
                result,
                source_name=source_name,
                conversion=conversion,
            )

    potassium = result.measurements.get("POTASSIUM")
    thorium = result.measurements.get("THORIUM")
    uranium = result.measurements.get("URANIUM")
    sgr = result.measurements.get("SGR")
    cgr = result.measurements.get("CGR")

    coefficients = dict(DEFAULT_API_COEFFICIENTS)
    if api_coefficients:
        for key, value in api_coefficients.items():
            canonical = str(key).strip().upper()
            if canonical not in coefficients:
                raise ValueError(f"Unknown API coefficient: {key!r}")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"API coefficient for {canonical} must be finite and non-negative"
                )
            coefficients[canonical] = numeric

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

    if thorium is not None and potassium is not None:
        add_derived(
            "TH_K_RATIO",
            _safe_ratio(thorium, potassium),
            "ppm/%",
            "Thorium to potassium ratio from processed spectral gamma ray",
        )
    if uranium is not None and thorium is not None:
        add_derived(
            "U_TH_RATIO",
            _safe_ratio(uranium, thorium),
            "ratio",
            "Uranium to thorium ratio from processed spectral gamma ray",
        )
    if uranium is not None and potassium is not None:
        add_derived(
            "U_K_RATIO",
            _safe_ratio(uranium, potassium),
            "ppm/%",
            "Uranium to potassium ratio from processed spectral gamma ray",
        )

    cgr_calc = None
    if potassium is not None and thorium is not None:
        cgr_calc = (
            coefficients["POTASSIUM"] * potassium + coefficients["THORIUM"] * thorium
        )
        add_derived(
            "CGR_CALC",
            cgr_calc,
            "API",
            "Calculated uranium-free gamma ray from potassium and thorium",
        )
        if cgr is not None:
            add_derived(
                "CGR_RESIDUAL",
                cgr - cgr_calc,
                "API",
                "Measured minus calculated uranium-free gamma ray",
            )

    sgr_calc = None
    if cgr_calc is not None and uranium is not None:
        sgr_calc = cgr_calc + coefficients["URANIUM"] * uranium
        add_derived(
            "SGR_CALC",
            sgr_calc,
            "API",
            "Calculated total spectral gamma ray from potassium, thorium, and uranium",
        )
        if sgr is not None:
            add_derived(
                "SGR_RESIDUAL",
                sgr - sgr_calc,
                "API",
                "Measured minus calculated total spectral gamma ray",
            )

    uranium_api = None
    if sgr is not None and cgr is not None:
        uranium_api = sgr - cgr
    elif uranium is not None:
        uranium_api = coefficients["URANIUM"] * uranium

    sgr_reference = sgr if sgr is not None else sgr_calc
    if uranium_api is not None:
        add_derived(
            "SGR_URANIUM_API",
            uranium_api,
            "API",
            "Uranium contribution to total spectral gamma ray",
        )
    if uranium_api is not None and sgr_reference is not None:
        uranium_fraction = _safe_ratio(np.maximum(uranium_api, 0.0), sgr_reference)
        add_derived(
            "SGR_URANIUM_FRACTION",
            uranium_fraction,
            "fraction",
            "Fraction of total spectral gamma ray attributed to uranium",
        )
        flag = np.full(uranium_fraction.shape, np.nan)
        finite = np.isfinite(uranium_fraction)
        flag[finite] = (uranium_fraction[finite] >= uranium_fraction_threshold).astype(
            float
        )
        add_derived(
            "SGR_URANIUM_EXCESS_FLAG",
            flag,
            "flag",
            "Screening flag for uranium-dominated total gamma ray",
        )

    return result


__all__ = [
    "DEFAULT_API_COEFFICIENTS",
    "SpectralGammaResult",
    "has_spectral_gamma_inputs",
    "interpret_spectral_gamma",
]
