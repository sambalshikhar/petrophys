"""
Main curve standardization module.

This module provides the CurveStandardizer class for standardizing
curve mnemonics and units across wells.
"""

import warnings
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.exceptions import DataQualityWarning

from petrophys.preprocessing.standardization.user_aliases import user_standard_name
from petrophys.preprocessing.standardization.alias_map import (
    get_aliases,
    get_standard_mnemonic,
    get_standard_unit,
    base_curve_mnemonic,
    ALIAS_MAP,
    NEUTRON_MATRIX_REFERENCE,
)
from petrophys.preprocessing.standardization.unit_converter import (
    CORE_SATURATION_CURVES,
    _benchmark_units,
    _conversion_factor,
    _converts_temperature,
    convert_curve_units,
    convert_units,
    standardize_core_saturation,
    standardize_units,
    get_conversion_factor,
)
from petrophys.model.units import PERCENT_SATURATION_UNITS, UnitConverter, temperature_unit

if TYPE_CHECKING:
    from petrophys.model.curve import Curve
    from petrophys.model.well import Well


CURVE_PLAUSIBLE_RANGES = {
    "DTC": (20.0, 300.0),
    "DTS": (20.0, 600.0),
    "DTE": (20.0, 600.0),
    "RHOB": (1.0, 3.5),
    "DRHO": (-1.0, 1.0),
    "NPHI": (-0.15, 1.0),
    "PHIN": (0.0, 0.6),
    "PHID": (-0.2, 0.8),
    "PHIS": (0.0, 0.8),
    "PHIT": (0.0, 0.8),
    "PHIE": (0.0, 0.8),
    "PHIE_CC": (0.0, 0.8),
    "PHI_SEC": (-0.2, 0.8),
    "VCL": (0.0, 1.0),
    "VCLAY": (0.0, 1.0),
    "VSH": (0.0, 1.0),
    "SW": (0.0, 1.0),
    "SO": (0.0, 1.0),
    "SWA": (0.0, 1.0),
    "SXO": (0.0, 1.0),
    "SWXOA": (0.0, 1.0),
    "SWIRR_APP": (0.0, 1.0),
    "BVW": (0.0, 1.0),
    "GR": (0.0, 500.0),
    "PEF": (0.0, 15.0),
    "CALI": (4.0, 40.0),
    "BS": (4.0, 40.0),
    "RDEP": (0.0, 1.0e5),
    "RMED": (0.0, 1.0e5),
    "RSHA": (0.0, 1.0e5),
    "RXO": (0.0, 1.0e5),
    "RW": (0.0, 10.0),
    "RMF": (0.0, 1000.0),
    "PERM": (0.0, 10000.0),
    "PERM_CC": (0.0, 10000.0),
    "TEMP": (32.0, 400.0),
    "TMPC": (32.0, 400.0),
    "VP_VS_RATIO": (0.5, 5.0),
    "VQUAR": (0.0, 1.0),
    "VCALC": (0.0, 1.0),
    "VDOLO": (0.0, 1.0),
    "VANHY": (0.0, 1.0),
    "NET_ROCK": (0.0, 1.0),
    "NET_RES": (0.0, 1.0),
    "NET_PAY": (0.0, 1.0),
    "RESFLAG": (0.0, 1.0),
    "SEALFLAG": (0.0, 1.0),
    "CROSSOVER_FLAG": (0.0, 1.0),
}

#: Curves whose plausible range above is in degF; a curve labelled in another
#: temperature scale is screened against the range converted to that scale.
TEMPERATURE_CURVES = {"TEMP", "TMPC"}

POROSITY_FRACTION_CURVES = {
    "NPHI",
    "PHIN",
    "PHID",
    "PHIS",
    "PHIT",
    "PHIE",
    "PHIE_CC",
    "PHI_SEC",
}


def _curve_standardization_priority(curve: "Curve") -> int:
    custom = getattr(getattr(curve, "metadata", None), "custom", {}) or {}
    try:
        return int(custom.get("standardization_priority", 0))
    except (TypeError, ValueError):
        return 0


#: Medium (60 in) array curves the RDEP alias passes over when the 90 in curve
#: of the same tool is present: the 90 in curve is the deep reading.
_SHALLOWER_ARRAY_DEEP_ALIASES = {"AT60": "AT90", "AO60": "AO90"}


def _unconfirmed_cala(mnemonic: str, curve: "Curve") -> bool:
    """CALA is also a voltage channel; require length units for a caliper alias.

    EDE-01 uses CALA in inches for its ultrasonic caliper. SLB's PWLS
    catalogue also uses CALA for calibration waveform amplitude.
    """
    return (
        base_curve_mnemonic(mnemonic) == "CALA"
        and get_conversion_factor(str(curve.unit or "").strip(), "in") is None
    )


def _differential_caliper(mnemonic: str, curve: "Curve") -> bool:
    """True for a DCAL holding a differential caliper (caliper minus bit size).

    DCAL names the density tool's caliper in some deliveries and the
    differential caliper in others. A median below 4 in, smaller than any
    hole, reads as the differential one, which is not a CALI. Not applied by
    the benchmark profiles.
    """
    if base_curve_mnemonic(mnemonic) != "DCAL" or reproducing_benchmark():
        return False
    values = np.asarray(curve.data, dtype=float)
    finite = values[np.isfinite(values) & (values > -900.0)]
    if finite.size == 0:
        return False
    unit = str(getattr(curve, "unit", "") or "").strip().lower()
    median = float(np.median(finite))
    if unit == "mm":
        median /= 25.4
    elif unit == "cm":
        median /= 2.54
    return median < 4.0


def add_standard_curve_aliases(
    well: "Well",
    standards: Optional[List[str]] = None,
    *,
    prefer_deepest_array: bool = True,
) -> Dict[str, str]:
    """Add canonical working aliases without removing vendor mnemonics.

    Curves marked as lower-quality or uncalibrated by an input reader are not
    promoted into canonical interpretation slots. With
    ``prefer_deepest_array``, RDEP takes an AT90/AO90 curve over the medium
    AT60/AO60 curve of the same tool whatever their sample counts.
    """
    requested = [name.upper() for name in (standards or list(ALIAS_MAP))]
    alias_map: Dict[str, str] = {}
    # The benchmark context profiles keep the sample-count choice between AT60 and AT90.
    prefer_deepest_array = prefer_deepest_array and not reproducing_benchmark()
    valid_bases = {
        base_curve_mnemonic(name)
        for name in well.curves
        if well.get(name) is not None and well.get(name).valid_count() > 0
    }

    for standard in requested:
        direct = well.get(standard)
        user_named = any(
            user_standard_name(well, name) == standard and name.upper() != standard for name in well.curves
        )
        if (
            direct is not None
            and direct.valid_count() > 0
            and _curve_standardization_priority(direct) >= 0
            and not user_named
        ):
            continue

        aliases = [standard, *get_aliases(standard)]
        alias_rank = {name.upper(): index for index, name in enumerate(aliases)}
        candidates = []
        for order, mnemonic in enumerate(well.curves):
            curve = well.get(mnemonic)
            if curve is None or curve.valid_count() == 0:
                continue
            priority = _curve_standardization_priority(curve)
            if priority < 0:
                continue
            base = base_curve_mnemonic(mnemonic)
            user = user_standard_name(well, mnemonic)
            if user is not None:
                if user != standard:
                    continue
                # The user's mapping outranks every alias.
                candidates.append(((priority, 2, 0, curve.valid_count(), 0, -order), mnemonic, curve))
                continue
            if base not in alias_rank:
                continue
            if standard == "CALI" and (
                _differential_caliper(mnemonic, curve) or _unconfirmed_cala(mnemonic, curve)
            ):
                continue
            shallower_array = (
                prefer_deepest_array
                and standard == "RDEP"
                and _SHALLOWER_ARRAY_DEEP_ALIASES.get(base) in valid_bases
            )
            score = (
                priority,
                int(base == standard),
                -int(shallower_array),
                curve.valid_count(),
                -alias_rank[base],
                -order,
            )
            candidates.append((score, mnemonic, curve))

        if not candidates:
            continue
        _, source_name, source_curve = max(candidates, key=lambda item: item[0])
        canonical = source_curve.copy()
        canonical.mnemonic = standard
        custom = dict(getattr(canonical.metadata, "custom", {}) or {})
        custom["working_alias_source"] = source_name
        canonical.metadata.custom = custom
        _note_neutron_matrix_reference(canonical, source_name, standard)
        well._curves[standard] = canonical
        alias_map[source_name] = standard

    if alias_map:
        well._parameters["_working_alias_map"] = alias_map
    return alias_map


def _maybe_convert_percent_porosity(data: np.ndarray, standard_name: str) -> np.ndarray:
    if standard_name.upper() not in POROSITY_FRACTION_CURVES:
        return data
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data
    median = float(np.nanmedian(finite))
    if median > 1.0:
        return data / 100.0
    return data


#: Vendor density and sonic porosities, often delivered in percent under a
#: blank label. They keep their own names and are read in v/v. Core
#: porosities are converted by :func:`~petrophys.interpretation.core_analysis.core_data_from_well`.
PERCENT_PRONE_POROSITY_CURVES = {"DPOR", "DPHI", "DPHZ", "PORD", "SPOR", "SPHI", "PORS"}

#: Unit labels of a fraction (lower case).
_FRACTION_UNITS = {"v/v", "frac", "fraction", "dec", "decimal", "m3/m3", "ft3/ft3"}


def _convert_percent_prone_porosity(curve: "Curve") -> None:
    """Convert a percent-prone porosity curve in percent to v/v, in place.

    The curve is in percent when its unit is a percent label and its values
    are not fractions (see :func:`_percent_label_holds_fractions`), or when
    its unit is not a fraction label and the median of its valid samples is
    above 1. Either way the unit becomes ``v/v``; a conversion is recorded in
    ``curve.metadata.custom["percent_porosity_conversion"]``.
    """
    values = np.asarray(curve.data, dtype=float)
    finite = values[np.isfinite(values) & (values > -900.0)]
    if finite.size == 0:
        return
    unit = str(curve.unit or "").strip().lower()
    median = float(np.median(finite))
    if unit in PERCENT_SATURATION_UNITS:
        if float(np.percentile(finite, 95)) <= 1.0 and median < 1.0:
            curve.unit = "v/v"
            return
    elif unit in _FRACTION_UNITS or median <= 1.0:
        return
    data = values.copy()
    data[np.isfinite(data) & (data <= -900.0)] = np.nan
    curve._data = data / 100.0
    curve.metadata.custom["percent_porosity_conversion"] = {
        "source_unit": curve.unit or "", "median_percent": median,
    }
    curve.unit = "v/v"


#: Share of a curve's samples that must be implausible before a
#: DataQualityWarning is raised; smaller shares are routine spike removal.
RANGE_SCREENING_WARNING_FRACTION = 0.5

_UNIT_SCALE_HINTS = (
    (1000.0, "1000x too large"),
    (100.0, "100x too large"),
    (0.001, "1000x too small"),
    (0.01, "100x too small"),
)


def _unit_scale_hint(values: np.ndarray, low: float, high: float) -> str:
    """Describe a scale error that would bring ``values`` into range, if any."""
    median = float(np.median(values))
    for factor, description in _UNIT_SCALE_HINTS:
        if low <= median / factor <= high:
            return f"values look {description}"
    return ""


def _range_screening_message(mnemonic: str, report: Dict[str, Any]) -> str:
    """Warning text for a curve whose range screening removed most samples.

    The report's ``source_channel`` and ``source_frame``, when given, name
    the channel and DLIS frame screened under ``mnemonic``.
    """
    source = report.get("source_channel")
    parts = [f"channel {source}"] if source and str(source).upper() != str(mnemonic).upper() else []
    if report.get("source_frame"):
        parts.append(f"frame {report['source_frame']}")
    label = f"{mnemonic} ({', '.join(parts)})" if parts else mnemonic
    message = (
        f"{label}: {report['masked']} of {report['finite']} samples are outside the "
        f"plausible {report['standard_curve']} range {report['low']:g} to {report['high']:g} "
        f"{report['unit'] or ''}".rstrip()
        + " and were removed"
    )
    if report.get("whole_curve") and report["finite"] > report["masked"]:
        message += f"; the other {report['finite'] - report['masked']} samples are left out too"
    if report["hint"]:
        message += f"; {report['hint']}, check the unit label {report['source_unit']!r}"
    return message + "."


def _check_sonic_scale(curve: "Curve", data: np.ndarray, source_unit: Optional[str]) -> None:
    """Warn when a compressional slowness looks like us/m values under a us/ft label.

    That is when its median and 5th percentile both lie outside
    :data:`SONIC_ROCK_BAND_US_FT` and inside it once divided by
    :data:`US_PER_M_IN_US_PER_FT`. The values are left unchanged; the finding
    is recorded in ``curve.metadata.custom["unit_scale_check"]``.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    low, high = SONIC_ROCK_BAND_US_FT
    median, p5 = float(np.median(finite)), float(np.percentile(finite, 5))

    def inside(value: float) -> bool:
        return low <= value <= high

    if inside(median) or inside(p5):
        return
    if not (inside(median / US_PER_M_IN_US_PER_FT) and inside(p5 / US_PER_M_IN_US_PER_FT)):
        return
    curve.metadata.custom["unit_scale_check"] = {
        "suspected_unit": "us/m",
        "median": median,
        "p5": p5,
        "factor": US_PER_M_IN_US_PER_FT,
    }
    warnings.warn(
        f"{curve.mnemonic}: median {median:.0f} and 5th percentile {p5:.0f} lie outside the "
        f"{low:g}-{high:g} us/ft of rock but inside it divided by {US_PER_M_IN_US_PER_FT}; the values "
        f"look like us/m labelled {source_unit!r}. They are not converted: relabel the curve us/m "
        f"(or divide it by {US_PER_M_IN_US_PER_FT}) if so.",
        DataQualityWarning,
        stacklevel=4,
    )


def _screen_standard_curve_values(
    curve: "Curve",
    standard_name: str,
    source_unit: Optional[str] = None,
    plausible_ranges: Optional[Dict[str, tuple]] = None,
    source_mnemonic: Optional[str] = None,
) -> "Curve":
    """Mask physically implausible standardized curve values.

    Masked samples are recorded in ``curve.metadata.custom["range_screening"]``.
    When most samples are implausible, which usually means a wrong unit
    label, a :class:`~petrophys.exceptions.DataQualityWarning` is raised and
    the whole curve is masked, so its in-range rest is not interpreted.
    ``plausible_ranges`` replaces :data:`CURVE_PLAUSIBLE_RANGES`.
    ``source_mnemonic`` names the curve the standard curve comes from, for
    the warning when the curve metadata gives no ``original_channel``.
    """
    name = standard_name.upper()
    historical = reproducing_benchmark()
    if name in PERCENT_PRONE_POROSITY_CURVES and not historical:
        _convert_percent_prone_porosity(curve)
    ranges = CURVE_PLAUSIBLE_RANGES if plausible_ranges is None else plausible_ranges
    limits = ranges.get(name)
    if limits is None:
        return curve

    data = np.asarray(curve.data, dtype=float).copy()
    data[np.isfinite(data) & (data <= -900.0)] = np.nan
    fractions = _maybe_convert_percent_porosity(data, name)
    if fractions is not data and not historical:
        # Percent values under a blank or other label, read as v/v.
        curve.unit = "v/v"
    data = fractions
    if name == "DTC" and not reproducing_benchmark():
        _check_sonic_scale(curve, data, source_unit if source_unit is not None else curve.unit)

    low, high = limits
    scale = temperature_unit(curve.unit) if name in TEMPERATURE_CURVES else None
    if scale in ("C", "K"):
        low, high = (float(UnitConverter.temperature(limit, "F", scale)) for limit in limits)
    finite = np.isfinite(data)
    invalid = finite & ((data < low) | (data > high))
    if np.any(invalid):
        masked = int(np.count_nonzero(invalid))
        total = int(np.count_nonzero(finite))
        # A scale hint only makes sense when most of the curve is implausible;
        # a few spikes of any size say nothing about the unit label.
        mostly_invalid = masked / total >= RANGE_SCREENING_WARNING_FRACTION
        hint = _unit_scale_hint(data[invalid], low, high) if mostly_invalid else ""
        report = {
            "standard_curve": name,
            "low": low,
            "high": high,
            "unit": curve.unit,
            "source_unit": source_unit if source_unit is not None else curve.unit,
            "masked": masked,
            "finite": total,
            "fraction": masked / total,
            "hint": hint,
        }
        # A mostly implausible curve has a wrong unit label or is another
        # measurement; its in-range rest is not trusted either.
        whole_curve = mostly_invalid and not historical
        report["whole_curve"] = whole_curve
        if not historical:
            # The channel (and DLIS frame) screened under the standard name.
            custom = curve.metadata.custom
            source = custom.get("original_channel") or source_mnemonic
            if source:
                report["source_channel"] = str(source)
            if custom.get("frame"):
                report["source_frame"] = str(custom["frame"])
        curve.metadata.custom["range_screening"] = report
        if mostly_invalid:
            warnings.warn(
                _range_screening_message(curve.mnemonic, report), DataQualityWarning, stacklevel=4
            )
        data[finite if whole_curve else invalid] = np.nan
        curve._data = data
    elif not np.array_equal(curve.data, data):
        curve._data = data
    if name in RESISTIVITY_CURVES and not reproducing_benchmark():
        _screen_resistivity_ceiling(curve)
    return curve


#: A resistivity whose maximum is at least this (ohm.m) and repeats in at least
#: RESISTIVITY_CEILING_MIN_SAMPLES samples is read as a ceiling, not a reading.
RESISTIVITY_CEILING_MIN_OHMM = 1000.0
RESISTIVITY_CEILING_MIN_SAMPLES = 3


def _screen_resistivity_ceiling(curve: "Curve") -> None:
    """Mask the samples of a resistivity curve that sit at a ceiling value.

    A tool saturated at its upper limit, or a resistivity computed as
    1000/conductivity and capped where the conductivity is zero, writes the
    same maximum value at every such sample (for example 100000 ohm.m). Those
    samples carry no resistivity. They are set to NaN and recorded in
    ``curve.metadata.custom["ceiling_screening"]``, with a
    :class:`~petrophys.exceptions.DataQualityWarning`.
    """
    data = np.asarray(curve.data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    ceiling = float(finite.max())
    at_ceiling = data == ceiling
    masked = int(np.count_nonzero(at_ceiling))
    if ceiling < RESISTIVITY_CEILING_MIN_OHMM or masked < RESISTIVITY_CEILING_MIN_SAMPLES:
        return
    data = data.copy()
    data[at_ceiling] = np.nan
    curve._data = data
    curve.metadata.custom["ceiling_screening"] = {
        "value": ceiling,
        "unit": curve.unit,
        "masked": masked,
        "finite": int(finite.size),
    }
    warnings.warn(
        _ceiling_screening_message(curve.mnemonic, curve.metadata.custom["ceiling_screening"]),
        DataQualityWarning,
        stacklevel=4,
    )


def _ceiling_screening_message(mnemonic: str, report: Dict[str, Any]) -> str:
    """Warning text for a resistivity curve whose ceiling samples were removed."""
    return (
        f"{mnemonic}: {report['masked']} of {report['finite']} samples sit at the curve's "
        f"maximum {report['value']:g} {report['unit'] or 'ohm.m'}, a tool or computation "
        "ceiling rather than a reading, and were removed."
    )


def _standard_conversion_factor(unit: str, standard_unit: str, standard_name: str) -> Optional[float]:
    """Conversion factor from ``unit`` to a standard curve's unit.

    Core saturation curves skip the extra unit spellings: their "frac" label
    often holds percent values, which :func:`standardize_core_saturation`
    detects, so it is not relabelled "v/v" here.
    """
    spellings = not _benchmark_units() and standard_name.upper() not in CORE_SATURATION_CURVES
    return _conversion_factor(unit, standard_unit, spellings=spellings)


#: Resistivity curves. A unit label with no conversion to ohm.m keeps such a
#: curve out of its standard slot, except a conductivity, which is inverted.
RESISTIVITY_CURVES = {"RDEP", "RMED", "RSHA", "RXO"}

#: Conductivity units (lower case) and the resistivity in ohm.m of a reading
#: of 1 in them: ohm.m = numerator / conductivity.
CONDUCTIVITY_UNITS = {
    "mmho/m": 1000.0,
    "mmhos/m": 1000.0,
    "ms/m": 1000.0,
    "mho/m": 1.0,
    "mhos/m": 1.0,
    "s/m": 1.0,
}

#: Compressional slowness from rock matrix (about 40 us/ft) to shale (about
#: 200 us/ft). A DTC whose median and 5th percentile both lie outside it,
#: but inside it once divided by :data:`US_PER_M_IN_US_PER_FT`, looks like
#: us/m values under a us/ft label.
SONIC_ROCK_BAND_US_FT = (40.0, 200.0)
US_PER_M_IN_US_PER_FT = 3.2808


def _resistivity_from_conductivity(data: np.ndarray, unit: str) -> Optional[np.ndarray]:
    """Resistivity (ohm.m) of conductivity ``data`` in ``unit``, or None for another unit.

    Readings at or below zero, which have no finite resistivity, are NaN.
    """
    numerator = CONDUCTIVITY_UNITS.get(str(unit).strip().lower().replace(" ", ""))
    if numerator is None:
        return None
    data = np.asarray(data, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(data > 0.0, numerator / data, np.nan)


def _percent_label_holds_fractions(data: np.ndarray, standard_name: str) -> bool:
    """True when a porosity curve under a percent label holds fractions.

    That is when the median of its valid samples is below 1 and its 95th
    percentile at most 1: a porosity in percent reads above 1 pu in most rock.
    """
    if standard_name.upper() not in POROSITY_FRACTION_CURVES:
        return False
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values) & (values > -900.0)]
    if values.size == 0:
        return False
    return float(np.median(values)) < 1.0 and float(np.percentile(values, 95)) <= 1.0


def _convert_to_standard_unit(
    new_curve: "Curve",
    source_unit: str,
    mnemonic: str,
    standard_name: str,
    standard_unit: str,
    *,
    relabel_equivalent_units: bool,
    historical: bool,
) -> bool:
    """Convert ``new_curve`` from ``source_unit`` to ``standard_unit`` in place.

    Outside the benchmark reproduction a unit with no conversion raises a
    :class:`~petrophys.exceptions.DataQualityWarning`: a resistivity curve in
    a conductivity unit is inverted to ohm.m; another resistivity curve is
    left unconverted and marked (``standardization_priority`` -1) to stay out
    of its standard slot; other curves are used unconverted. A porosity curve
    under a percent label whose values are fractions is used as fractions,
    with a warning. Returns False when the curve stays out of the standard slot.
    """
    if standard_name == "DEPTH" and not historical:
        # A depth curve (MD, DEPT) stays in its own unit, as the depth index does.
        return True
    kind = get_standard_mnemonic(standard_name)
    factor = _standard_conversion_factor(source_unit, standard_unit, standard_name)
    if factor is not None:
        if factor == 0.01 and not historical and _percent_label_holds_fractions(new_curve._data, kind):
            values = np.asarray(new_curve._data, dtype=float)
            median = float(np.nanmedian(values[values > -900.0]))
            warnings.warn(
                f"{mnemonic}: unit {source_unit!r} is percent but the values (median {median:.3g}) "
                f"are fractions; they are used as {standard_unit} without dividing by 100. "
                "Check the unit label.",
                DataQualityWarning,
                stacklevel=3,
            )
            new_curve.unit = standard_unit
        elif factor != 1.0:
            new_curve._data = new_curve._data * factor
            new_curve.unit = standard_unit
        elif relabel_equivalent_units:
            new_curve.unit = standard_unit
    elif _converts_temperature(source_unit, standard_unit):
        # e.g. degC to F. Labels of the standard scale, such
        # as degF, are left as written.
        new_curve._data = convert_units(new_curve._data, source_unit, standard_unit)
        new_curve.unit = standard_unit
    elif temperature_unit(source_unit) is not None and temperature_unit(source_unit) == temperature_unit(standard_unit):
        # Another spelling of the standard temperature scale (degF, DEGF, deg F):
        # already in the standard unit, and the label is left as written.
        pass
    elif not historical:
        resistivity = (
            _resistivity_from_conductivity(new_curve._data, source_unit)
            if kind in RESISTIVITY_CURVES
            else None
        )
        if resistivity is not None:
            new_curve._data = resistivity
            new_curve.unit = standard_unit
            message = (
                f"{mnemonic}: unit {source_unit!r} is a conductivity; converted to resistivity "
                f"in {standard_unit} for {kind}."
            )
        elif kind in RESISTIVITY_CURVES:
            new_curve.metadata.custom["standardization_priority"] = -1
            new_curve.metadata.custom["unconvertible_unit"] = source_unit
            message = (
                f"{mnemonic}: unit {source_unit!r} cannot be converted to {standard_unit!r}; "
                f"the curve is kept as {_unconverted_name(mnemonic, standard_name)} and not used "
                f"as {kind}."
            )
        else:
            message = (
                f"{mnemonic}: unit {source_unit!r} cannot be converted to {standard_unit!r}; "
                f"the values are used as {kind} unconverted. Check the unit label."
            )
        warnings.warn(message, DataQualityWarning, stacklevel=3)
        return resistivity is not None or kind not in RESISTIVITY_CURVES
    return True


def _unconverted_name(mnemonic: str, standard_name: str) -> str:
    """Name of a curve kept out of the standard slot ``standard_name``: its own, or <name>_UNCONVERTED."""
    return f"{mnemonic}_UNCONVERTED" if mnemonic == standard_name else mnemonic


def _note_neutron_matrix_reference(curve: "Curve", source_mnemonic: str, standard_name: str) -> None:
    """Record the matrix of a neutron alias read on another matrix than limestone.

    The reference goes into the curve description and
    ``metadata.custom["neutron_matrix_reference"]``, with a
    :class:`~petrophys.exceptions.DataQualityWarning` the first time.
    """
    reference = NEUTRON_MATRIX_REFERENCE.get(base_curve_mnemonic(source_mnemonic))
    if standard_name != "NPHI" or reference is None or reproducing_benchmark():
        return
    custom = curve.metadata.custom
    if custom.get("neutron_matrix_reference"):
        return
    custom["neutron_matrix_reference"] = reference
    from petrophys.interpretation.porosity.neutron_porosity import standardize_nphi

    values = np.asarray(curve.data, dtype=float)
    converted = standardize_nphi(values, reference, "limestone")
    offset = float(np.nanmedian(converted - values)) if np.isfinite(values).any() else 0.0
    curve._data = converted
    custom["neutron_matrix_conversion"] = f"{reference} to limestone scale, {offset:+.2f} v/v"
    note = (f"{reference}-referenced neutron porosity ({base_curve_mnemonic(source_mnemonic)}), "
            f"converted to the limestone scale ({offset:+.2f} v/v)")
    description = str(curve.metadata.description or "").strip()
    curve.metadata.description = f"{description}; {note}" if description else note
    warnings.warn(
        f"{source_mnemonic} is a {reference}-referenced neutron porosity and is used as NPHI, which "
        f"interpretation reads on the limestone scale: it was converted by the first-order offset "
        f"{offset:+.2f} v/v (the chart offset varies by about 0.01-0.02 with porosity).",
        DataQualityWarning,
        stacklevel=3,
    )


class CurveStandardizer:
    """
    Standardizes curve mnemonics and units.

    This class provides functionality to:
    - Map curve aliases to standard mnemonics
    - Convert units to standard units
    - Handle multiple variants of the same curve type

    Parameters
    ----------
    rename_curves : bool, optional
        Whether to rename curves to standard mnemonics (default: True)
    convert_units : bool, optional
        Whether to convert to standard units (default: True)
    preserve_originals : bool, optional
        Keep original curves with _ORIG suffix (default: False)
    relabel_equivalent_units : bool, optional
        Write the standard unit label on a curve whose unit is another
        spelling of it, with a conversion factor of 1, e.g. ``gAPI`` for
        ``API`` (default: True). False leaves such labels as written.
    plausible_ranges : dict, optional
        ``{standard mnemonic: (low, high)}`` limits of the implausible-value
        screening, in the standard units; default
        :data:`CURVE_PLAUSIBLE_RANGES`.

    Examples
    --------
    >>> standardizer = CurveStandardizer()
    >>> well = standardizer.standardize(well)
    """

    def __init__(
        self,
        rename_curves: bool = True,
        convert_units: bool = True,
        preserve_originals: bool = False,
        relabel_equivalent_units: bool = True,
        plausible_ranges: Optional[Dict[str, tuple]] = None,
    ):
        self.rename_curves = rename_curves
        self.convert_units = convert_units
        self.preserve_originals = preserve_originals
        self.plausible_ranges = plausible_ranges
        #: Rewrite a curve's unit label to the standard one even when the two
        #: are equivalent (conversion factor 1), e.g. ``m3/m3`` to ``v/v``.
        #: When False such labels are left as the source wrote them.
        self.relabel_equivalent_units = relabel_equivalent_units

        # Custom aliases (in addition to default)
        self._custom_aliases: Dict[str, List[str]] = {}

    def add_alias(self, standard: str, alias: str):
        """
        Add a custom alias mapping.

        Parameters
        ----------
        standard : str
            Standard mnemonic
        alias : str
            Alias to map to standard
        """
        standard = standard.upper()
        alias = alias.upper()

        if standard not in self._custom_aliases:
            self._custom_aliases[standard] = []
        if alias not in self._custom_aliases[standard]:
            self._custom_aliases[standard].append(alias)

    def get_standard_name(self, mnemonic: str) -> str:
        """
        Get the standard mnemonic for a curve.

        Parameters
        ----------
        mnemonic : str
            Input mnemonic

        Returns
        -------
        str
            Standard mnemonic
        """
        mnemonic = mnemonic.upper()

        # Check custom aliases first
        for standard, aliases in self._custom_aliases.items():
            if mnemonic in aliases or mnemonic == standard:
                return standard

        # Use default mapping
        return get_standard_mnemonic(mnemonic)

    def standardize(self, well: "Well") -> "Well":
        """
        Standardize all curves in a well.

        Parameters
        ----------
        well : Well
            Well object to standardize

        Returns
        -------
        Well
            Well with standardized curves
        """
        from petrophys.model.curve import Curve, CurveMetadata
        from petrophys.model.data_types import DataType

        # Track renamed curves to avoid conflicts
        historical = reproducing_benchmark()
        new_curves = {}
        rename_map = {}
        #: Original mnemonic of the curve holding each standard name.
        slot_source = {}

        for mnemonic, curve in well._curves.items():
            # Get standard name
            user_named = user_standard_name(well, mnemonic)
            standard_name = user_named or self.get_standard_name(mnemonic)
            if standard_name == "CALI" and _unconfirmed_cala(mnemonic, curve):
                new_curves[mnemonic] = curve.copy()
                continue
            if standard_name == "CALI" and _differential_caliper(mnemonic, curve):
                # Differential or unconfirmed calipers must keep their own names.
                standard_name = mnemonic
            source_priority = _curve_standardization_priority(curve)
            eligible_for_standard_slot = source_priority >= 0
            own_name = mnemonic

            # Preserve original if requested
            if self.preserve_originals and standard_name != mnemonic:
                new_curves[f"{mnemonic}_ORIG"] = curve.copy()

            # Create new curve
            new_curve = curve.copy()

            # Rename if needed
            if (
                self.rename_curves
                and eligible_for_standard_slot
                and standard_name != mnemonic
            ):
                new_curve.mnemonic = standard_name
                rename_map[mnemonic] = standard_name

            # Convert units if needed
            if self.convert_units and standard_name in CORE_SATURATION_CURVES and not historical:
                # Core saturations: percent values under a "frac" or blank
                # label are detected and converted to v/v, with a warning.
                try:
                    standardize_core_saturation(new_curve, inplace=True)
                except ValueError as exc:
                    warnings.warn(
                        f"{mnemonic}: {exc}; the curve is left unconverted.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif self.convert_units:
                standard_unit = get_standard_unit(standard_name)
                if standard_unit and curve.unit and not _convert_to_standard_unit(
                    new_curve,
                    curve.unit,
                    mnemonic,
                    standard_name,
                    standard_unit,
                    relabel_equivalent_units=self.relabel_equivalent_units,
                    historical=historical,
                ):
                    # Unconvertible resistivity unit: the curve keeps its own
                    # name, and a curve that already has the standard name
                    # moves aside to <name>_UNCONVERTED.
                    eligible_for_standard_slot = False
                    rename_map.pop(mnemonic, None)
                    own_name = _unconverted_name(mnemonic, standard_name)
                    if own_name != mnemonic:
                        rename_map[mnemonic] = own_name
                    new_curve.mnemonic = own_name

            _note_neutron_matrix_reference(new_curve, mnemonic, standard_name)
            _screen_standard_curve_values(
                new_curve, standard_name, curve.unit, self.plausible_ranges, source_mnemonic=mnemonic
            )

            # Handle duplicate standard names: the incoming curve takes the
            # standard name only with a higher source priority (DLIS channel
            # quality) or, at equal priority, more valid samples. The other
            # curve keeps its own original mnemonic; no curve is dropped.
            if (
                standard_name in new_curves
                and self.rename_curves
                and eligible_for_standard_slot
            ):
                existing = new_curves[standard_name]
                existing_priority = _curve_standardization_priority(existing)
                if user_named and user_standard_name(well, slot_source[standard_name]) != standard_name:
                    # The user's mapping outranks a curve that is merely named so.
                    existing_priority = min(existing_priority, source_priority - 1)
                wins = source_priority > existing_priority or (
                    source_priority == existing_priority
                    and new_curve.valid_count() > existing.valid_count()
                )
                if historical:
                    # Apply the selected benchmark context profile's calculation rules.
                    if wins:
                        new_curves[mnemonic] = existing
                        new_curves[standard_name] = new_curve
                    continue
                if wins:
                    kept, kept_name = existing, slot_source[standard_name]
                    new_curves[standard_name] = new_curve
                    slot_source[standard_name] = mnemonic
                else:
                    kept, kept_name = new_curve, mnemonic
                if kept_name == standard_name:
                    # The losing curve already carried the standard name.
                    kept_name = f"{standard_name}_2"
                    while kept_name in new_curves:
                        kept_name += "_2"
                kept.mnemonic = kept_name
                rename_map.pop(kept_name, None)
                new_curves[kept_name] = kept
                warnings.warn(
                    f"{kept_name} and {slot_source[standard_name]} both map to {standard_name}; "
                    f"{slot_source[standard_name]} is used as {standard_name} and {kept_name} keeps its own name.",
                    DataQualityWarning,
                    stacklevel=2,
                )
            else:
                target_name = (
                    standard_name
                    if self.rename_curves and eligible_for_standard_slot
                    else own_name
                )
                new_curves[target_name] = new_curve
                slot_source[target_name] = mnemonic

        # Update well curves
        well._curves = new_curves

        # Store rename map in well metadata
        well._parameters["_rename_map"] = rename_map

        return well

    def standardize_curve(
        self,
        curve: "Curve",
        rename: Optional[bool] = None,
        convert: Optional[bool] = None
    ) -> "Curve":
        """
        Standardize a single curve.

        Parameters
        ----------
        curve : Curve
            Curve to standardize
        rename : bool, optional
            Override rename_curves setting
        convert : bool, optional
            Override convert_units setting

        Returns
        -------
        Curve
            Standardized curve
        """
        rename = rename if rename is not None else self.rename_curves
        convert = convert if convert is not None else self.convert_units

        new_curve = curve.copy()

        if _unconfirmed_cala(curve.mnemonic, curve):
            return new_curve

        if rename:
            standard_name = self.get_standard_name(curve.mnemonic)
            if standard_name == "CALI" and _differential_caliper(curve.mnemonic, curve):
                # Differential or unconfirmed calipers must keep their own names.
                standard_name = curve.mnemonic
            new_curve.mnemonic = standard_name
        else:
            standard_name = curve.mnemonic

        if convert:
            standard_unit = get_standard_unit(standard_name)
            if standard_unit and curve.unit and not _convert_to_standard_unit(
                new_curve,
                curve.unit,
                curve.mnemonic,
                standard_name,
                standard_unit,
                relabel_equivalent_units=True,
                historical=reproducing_benchmark(),
            ):
                # Unconvertible resistivity unit: the curve keeps its own name.
                new_curve.mnemonic = _unconverted_name(curve.mnemonic, standard_name)

        _screen_standard_curve_values(
            new_curve, standard_name, curve.unit, self.plausible_ranges, source_mnemonic=curve.mnemonic
        )

        return new_curve

    def find_curve(
        self,
        well: "Well",
        curve_type: str
    ) -> Optional["Curve"]:
        """
        Find a curve of specified type in well.

        Parameters
        ----------
        well : Well
            Well to search
        curve_type : str
            Standard curve type (e.g., "RDEP", "RHOB")

        Returns
        -------
        Curve or None
            Found curve, or None
        """
        curve_type = curve_type.upper()

        # Direct match
        if curve_type in well._curves:
            return well._curves[curve_type]

        # Search aliases
        aliases = get_aliases(curve_type)
        for alias in aliases:
            if alias in well._curves:
                curve = well._curves[alias]
                if curve_type == "CALI" and (
                    _unconfirmed_cala(alias, curve) or _differential_caliper(alias, curve)
                ):
                    continue
                return curve

        # Search custom aliases
        custom = self._custom_aliases.get(curve_type, [])
        for alias in custom:
            if alias in well._curves:
                return well._curves[alias]

        return None

    def report(self, well: "Well") -> Dict[str, Any]:
        """
        Generate standardization report for a well.

        Parameters
        ----------
        well : Well
            Well to analyze

        Returns
        -------
        dict
            Report with standardization recommendations
        """
        report = {
            "total_curves": len(well.curves),
            "recognized": [],
            "unrecognized": [],
            "duplicates": [],
            "unit_conversions": [],
        }

        # Track recognized types
        found_types = {}

        for mnemonic in well.curves:
            standard = self.get_standard_name(mnemonic)
            if standard == "CALI" and _unconfirmed_cala(mnemonic, well._curves[mnemonic]):
                report["unrecognized"].append(mnemonic)
                continue

            if standard != mnemonic:
                report["recognized"].append({
                    "original": mnemonic,
                    "standard": standard
                })
            else:
                # Check if it's in any alias list
                is_standard = standard in ALIAS_MAP
                if not is_standard:
                    report["unrecognized"].append(mnemonic)
                else:
                    report["recognized"].append({
                        "original": mnemonic,
                        "standard": standard
                    })

            # Track for duplicates
            if standard in found_types:
                report["duplicates"].append({
                    "type": standard,
                    "curves": [found_types[standard], mnemonic]
                })
            else:
                found_types[standard] = mnemonic

            # Check unit conversion
            curve = well._curves[mnemonic]
            if curve.unit:
                standard_unit = get_standard_unit(standard)
                if standard_unit and curve.unit.lower() != standard_unit.lower():
                    factor = get_conversion_factor(curve.unit, standard_unit)
                    if factor is not None or _converts_temperature(curve.unit, standard_unit):
                        report["unit_conversions"].append({
                            "curve": mnemonic,
                            "from_unit": curve.unit,
                            "to_unit": standard_unit
                        })

        return report


def standardize_curves(
    well: "Well",
    rename_curves: bool = True,
    convert_units: bool = True,
    preserve_originals: bool = False,
    relabel_equivalent_units: bool = True,
    plausible_ranges: Optional[Dict[str, tuple]] = None,
) -> "Well":
    """
    Standardize all curves in a well.

    Convenience function for CurveStandardizer.

    Parameters
    ----------
    well : Well
        Well to standardize
    rename_curves : bool, optional
        Rename to standard mnemonics (default: True)
    convert_units : bool, optional
        Convert to standard units (default: True)
    preserve_originals : bool, optional
        Keep original curves (default: False)
    relabel_equivalent_units : bool, optional
        Write the standard unit label on a curve whose unit is another
        spelling of it, e.g. ``gAPI`` for ``API`` (default: True)
    plausible_ranges : dict, optional
        Limits of the implausible-value screening; default
        :data:`CURVE_PLAUSIBLE_RANGES`.

    Returns
    -------
    Well
        Standardized well

    Examples
    --------
    >>> from petrophys.preprocessing import standardize_curves
    >>> well = standardize_curves(well)
    """
    standardizer = CurveStandardizer(
        rename_curves=rename_curves,
        convert_units=convert_units,
        preserve_originals=preserve_originals,
        relabel_equivalent_units=relabel_equivalent_units,
        plausible_ranges=plausible_ranges,
    )
    return standardizer.standardize(well)
