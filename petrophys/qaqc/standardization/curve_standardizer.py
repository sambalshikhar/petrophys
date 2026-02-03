"""
Main curve standardization module.

This module provides the CurveStandardizer class for standardizing
curve mnemonics and units across wells.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from petrophys.qaqc.standardization.alias_map import (
    get_standard_mnemonic,
    get_standard_unit,
    ALIAS_MAP,
)
from petrophys.qaqc.standardization.unit_converter import (
    convert_curve_units,
    standardize_units,
    get_conversion_factor,
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
        from petrophys.core.curve import Curve, CurveMetadata
        from petrophys.core.data_types import DataType

        # Track renamed curves to avoid conflicts
        new_curves = {}
        rename_map = {}

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
                    if factor is not None and factor != 1.0:
                        new_curve._data = new_curve._data * factor
                        new_curve.unit = standard_unit

            # Handle duplicate standard names (keep first or best quality)
            if standard_name in new_curves and self.rename_curves:
                # Check which has more valid data
                existing = new_curves[standard_name]
                if new_curve.valid_count() > existing.valid_count():
                    # Move existing to original name
                    new_curves[mnemonic] = existing
                    new_curves[standard_name] = new_curve
                # else keep existing
            else:
                target_name = standard_name if self.rename_curves else mnemonic
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
                    new_curve._data = new_curve._data * factor
                    new_curve.unit = standard_unit

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
    >>> from petrophys.qaqc import standardize_curves
    >>> well = standardize_curves(well)
    """
    standardizer = CurveStandardizer(
        rename_curves=rename_curves,
        convert_units=convert_units,
        preserve_originals=preserve_originals
    )
    return standardizer.standardize(well)
