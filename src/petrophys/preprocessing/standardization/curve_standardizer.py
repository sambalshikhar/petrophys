"""
Main curve standardization module.

This module provides the CurveStandardizer class for standardizing
curve mnemonics and units across wells.
"""

import warnings
from typing import Optional, List, Dict, Any
import numpy as np

from petrophys.preprocessing.standardization.alias_map import (
    get_standard_mnemonic,
    get_standard_unit,
    ALIAS_MAP,
)
from petrophys.preprocessing.standardization.unit_converter import (
    convert_curve_units,
    standardize_units,
    get_conversion_factor,
)


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


def _screen_standard_curve_values(curve: "Curve", standard_name: str) -> "Curve":
    """Mask physically implausible standardized curve values."""
    name = standard_name.upper()
    limits = CURVE_PLAUSIBLE_RANGES.get(name)
    if limits is None:
        return curve

    data = np.asarray(curve.data, dtype=float).copy()
    data[np.isfinite(data) & (data <= -900.0)] = np.nan
    data = _maybe_convert_percent_porosity(data, name)

    low, high = limits
    invalid = np.isfinite(data) & ((data < low) | (data > high))
    if np.any(invalid):
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

    Examples
    --------
    >>> standardizer = CurveStandardizer()
    >>> well = standardizer.standardize(well)
    """

    def __init__(
        self,
        rename_curves: bool = True,
        convert_units: bool = True,
        preserve_originals: bool = False
    ):
        self.rename_curves = rename_curves
        self.convert_units = convert_units
        self.preserve_originals = preserve_originals

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
        # Original mnemonic of the curve currently occupying each standard slot,
        # so a collision loser can be parked under its own name (never dropped,
        # never stored under another curve's mnemonic).
        origin_of = {}

        def parking_name(origin: str, standard: str) -> str:
            """Return a free non-standard slot for a collision loser."""
            if origin != standard and origin not in new_curves:
                return origin
            base = f"{origin}_ALT"
            candidate = base
            suffix = 2
            while candidate == standard or candidate in new_curves:
                candidate = f"{base}_{suffix}"
                suffix += 1
            return candidate

        for mnemonic, curve in well._curves.items():
            # Get standard name
            standard_name = self.get_standard_name(mnemonic)

            # Preserve original if requested
            if self.preserve_originals and standard_name != mnemonic:
                new_curves[f"{mnemonic}_ORIG"] = curve.copy()

            # Create new curve
            new_curve = curve.copy()

            # Rename if needed
            if self.rename_curves and standard_name != mnemonic:
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

            _screen_standard_curve_values(new_curve, standard_name)

            # Handle duplicate standard names. Both curves always survive:
            # the winner takes the standard slot, the loser keeps its own
            # original mnemonic. A curve natively named like the standard
            # (e.g. a real RDEP alongside an ILD alias) takes the slot.
            if standard_name in new_curves and self.rename_curves:
                existing = new_curves[standard_name]
                existing_origin = origin_of.get(standard_name, standard_name)
                incoming_is_native = mnemonic == standard_name
                existing_custom = getattr(existing.metadata, "custom", {}) or {}
                incoming_custom = getattr(new_curve.metadata, "custom", {}) or {}
                existing_priority = int(existing_custom.get("standardization_priority", 0))
                incoming_priority = int(incoming_custom.get("standardization_priority", 0))
                incoming_wins = (
                    incoming_priority > existing_priority
                    or (
                        incoming_priority == existing_priority
                        and (
                            incoming_is_native
                            or new_curve.valid_count() > existing.valid_count()
                        )
                    )
                )
                if incoming_wins:
                    # Incoming takes the slot; park the previous occupant
                    # back under its own original mnemonic.
                    parked_name = parking_name(existing_origin, standard_name)
                    existing.mnemonic = parked_name
                    new_curves[parked_name] = existing
                    rename_map.pop(existing_origin, None)
                    new_curves[standard_name] = new_curve
                    origin_of[standard_name] = mnemonic
                    slot_holder, parked = mnemonic, parked_name
                else:
                    # Existing keeps the slot; park the incoming curve under
                    # its own original mnemonic.
                    parked_name = parking_name(mnemonic, standard_name)
                    new_curve.mnemonic = parked_name
                    new_curves[parked_name] = new_curve
                    rename_map.pop(mnemonic, None)
                    slot_holder, parked = existing_origin, parked_name
                warnings.warn(
                    f"Curves '{existing_origin}' and '{mnemonic}' both standardize "
                    f"to '{standard_name}'; '{slot_holder}' takes the "
                    f"'{standard_name}' slot and '{parked}' keeps its original name."
                )
            else:
                target_name = standard_name if self.rename_curves else mnemonic
                new_curves[target_name] = new_curve
                if self.rename_curves:
                    origin_of[standard_name] = mnemonic

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

        _screen_standard_curve_values(new_curve, standard_name)

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
                    if factor is not None:
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
    preserve_originals: bool = False
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
        preserve_originals=preserve_originals
    )
    return standardizer.standardize(well)
