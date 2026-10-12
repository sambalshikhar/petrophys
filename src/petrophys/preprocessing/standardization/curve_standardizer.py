"""
Main curve standardization module.

This module provides the CurveStandardizer class for standardizing
curve mnemonics and units across wells.
"""

import warnings
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import numpy as np

from petrophys.exceptions import DataQualityWarning

from petrophys.preprocessing.standardization.alias_map import (
    get_standard_mnemonic,
    get_standard_unit,
    base_curve_mnemonic,
    ALIAS_MAP,
)
from petrophys.preprocessing.standardization.unit_converter import (
    _converts_temperature,
    convert_curve_units,
    convert_units,
    standardize_units,
    get_conversion_factor,
)
from petrophys.model.units import UnitConverter, temperature_unit

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


def add_standard_curve_aliases(
    well: "Well",
    standards: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Add canonical working aliases without removing vendor mnemonics.

    Curves marked as lower-quality or uncalibrated by an input reader are not
    promoted into canonical interpretation slots.
    """
    requested = [name.upper() for name in (standards or list(ALIAS_MAP))]
    alias_map: Dict[str, str] = {}

    for standard in requested:
        direct = well.get(standard)
        if (
            direct is not None
            and direct.valid_count() > 0
            and _curve_standardization_priority(direct) >= 0
        ):
            continue

        aliases = [standard, *ALIAS_MAP.get(standard, [])]
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
            if base not in alias_rank:
                continue
            score = (
                priority,
                int(base == standard),
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
    """Warning text for a curve whose range screening removed most samples."""
    message = (
        f"{mnemonic}: {report['masked']} of {report['finite']} samples are outside the "
        f"plausible {report['standard_curve']} range {report['low']:g} to {report['high']:g} "
        f"{report['unit'] or ''}".rstrip()
        + " and were removed"
    )
    if report["hint"]:
        message += f"; {report['hint']}, check the unit label {report['source_unit']!r}"
    return message + "."


def _screen_standard_curve_values(
    curve: "Curve",
    standard_name: str,
    source_unit: Optional[str] = None,
    plausible_ranges: Optional[Dict[str, tuple]] = None,
) -> "Curve":
    """Mask physically implausible standardized curve values.

    Masked samples are recorded in ``curve.metadata.custom["range_screening"]``.
    When most samples are implausible, which usually means a wrong unit
    label, a :class:`~petrophys.exceptions.DataQualityWarning` is raised.
    ``plausible_ranges`` replaces :data:`CURVE_PLAUSIBLE_RANGES`.
    """
    name = standard_name.upper()
    ranges = CURVE_PLAUSIBLE_RANGES if plausible_ranges is None else plausible_ranges
    limits = ranges.get(name)
    if limits is None:
        return curve

    data = np.asarray(curve.data, dtype=float).copy()
    data[np.isfinite(data) & (data <= -900.0)] = np.nan
    data = _maybe_convert_percent_porosity(data, name)

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
        curve.metadata.custom["range_screening"] = report
        if masked / total >= RANGE_SCREENING_WARNING_FRACTION:
            warnings.warn(
                _range_screening_message(curve.mnemonic, report), DataQualityWarning, stacklevel=4
            )
        data[invalid] = np.nan
        curve._data = data
    elif not np.array_equal(curve.data, data):
        curve._data = data
    return curve


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
        new_curves = {}
        rename_map = {}

        for mnemonic, curve in well._curves.items():
            # Get standard name
            standard_name = self.get_standard_name(mnemonic)
            source_priority = _curve_standardization_priority(curve)
            eligible_for_standard_slot = source_priority >= 0

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
            if self.convert_units:
                standard_unit = get_standard_unit(standard_name)
                if standard_unit and curve.unit:
                    factor = get_conversion_factor(curve.unit, standard_unit)
                    if factor is not None:
                        if factor != 1.0:
                            new_curve._data = new_curve._data * factor
                            new_curve.unit = standard_unit
                        elif self.relabel_equivalent_units:
                            new_curve.unit = standard_unit
                    elif _converts_temperature(curve.unit, standard_unit):
                        # e.g. degC to F. Labels of the standard scale, such
                        # as degF, are left as written.
                        new_curve._data = convert_units(new_curve._data, curve.unit, standard_unit)
                        new_curve.unit = standard_unit

            _screen_standard_curve_values(new_curve, standard_name, curve.unit, self.plausible_ranges)

            # Handle duplicate standard names: the incoming curve replaces the
            # occupant only with a higher source priority (DLIS channel
            # quality) or, at equal priority, more valid samples; the occupant
            # is then kept under the incoming curve's mnemonic. Otherwise the
            # incoming curve is dropped.
            if (
                standard_name in new_curves
                and self.rename_curves
                and eligible_for_standard_slot
            ):
                existing = new_curves[standard_name]
                existing_priority = _curve_standardization_priority(existing)
                if source_priority > existing_priority or (
                    source_priority == existing_priority
                    and new_curve.valid_count() > existing.valid_count()
                ):
                    new_curves[mnemonic] = existing
                    new_curves[standard_name] = new_curve
            else:
                target_name = (
                    standard_name
                    if self.rename_curves and eligible_for_standard_slot
                    else mnemonic
                )
                new_curves[target_name] = new_curve

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

        if rename:
            standard_name = self.get_standard_name(curve.mnemonic)
            new_curve.mnemonic = standard_name
        else:
            standard_name = curve.mnemonic

        if convert:
            standard_unit = get_standard_unit(standard_name)
            if standard_unit and curve.unit:
                factor = get_conversion_factor(curve.unit, standard_unit)
                if factor is not None:
                    if factor != 1.0:
                        new_curve._data = new_curve._data * factor
                    new_curve.unit = standard_unit
                elif _converts_temperature(curve.unit, standard_unit):
                    new_curve._data = convert_units(new_curve._data, curve.unit, standard_unit)
                    new_curve.unit = standard_unit

        _screen_standard_curve_values(new_curve, standard_name, curve.unit, self.plausible_ranges)

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
        aliases = ALIAS_MAP.get(curve_type, [])
        for alias in aliases:
            if alias in well._curves:
                return well._curves[alias]

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
