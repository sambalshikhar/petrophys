"""
Export functionality for well log data.

This module provides functions to export Well data to various
file formats including LAS, CSV, and Excel.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import numpy as np

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


ML_IMPUTATION_NOTES_PARAMETER = "ML_IMPUTATION_NOTES"


def _export_well_other_comments(las, well: "Well") -> None:
    """Write workflow comments into the LAS ~Other section."""
    lines = []
    existing_other = ""
    if hasattr(well, "get_parameter"):
        existing_other = well.get_parameter("LAS_OTHER", "") or ""
    if existing_other:
        lines.extend(str(existing_other).splitlines())

    notes = []
    if hasattr(well, "get_parameter"):
        notes = well.get_parameter(ML_IMPUTATION_NOTES_PARAMETER, []) or []
    if isinstance(notes, str):
        notes = [notes]
    for note in notes:
        text = str(note).strip()
        if text:
            lines.append(text)

    if lines:
        las.other = "\n".join(lines)


def _export_well_parameters(las, well: "Well") -> None:
    """Preserve Well parameters in the LAS ~Params section."""
    parameters = getattr(well, "_parameters", {})
    if not parameters:
        return

    for name, value in parameters.items():
        if not isinstance(name, str):
            continue
        if "__" in name or name.startswith("_"):
            continue
        if name in {ML_IMPUTATION_NOTES_PARAMETER, "LAS_OTHER"}:
            continue

        unit = parameters.get(f"{name}__UNIT", "")
        descr = parameters.get(f"{name}__DESCR", "")

        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = ""

        las.params.append(
            lasio.HeaderItem(
                mnemonic=name,
                unit=unit or "",
                value=value,
                descr=descr or "",
            )
        )


def export_las(
    well: "Well",
    filepath: Union[str, Path],
    version: float = 2.0,
    wrap: bool = False,
    curves: Optional[List[str]] = None,
    *,
    extra_parameters: Optional[List[Dict[str, Any]]] = None,
    other: Optional[str] = None,
    **kwargs
):
    """
    Export Well to LAS file.

    Parameters
    ----------
    well : Well
        Well object to export
    filepath : str or Path
        Output file path
    version : float, optional
        LAS version (1.2 or 2.0, default: 2.0)
    wrap : bool, optional
        Whether to wrap long lines (default: False)
    curves : list of str, optional
        Specific curves to export (default: all)
    extra_parameters : list of dict, optional
        Additional ``~Params`` entries, each a mapping with ``mnemonic``,
        ``value`` and optional ``unit`` and ``descr`` keys (e.g. the settings
        an interpretation used).
    other : str, optional
        Text appended to the ``~Other`` section.
    **kwargs
        Additional arguments passed to lasio's LASFile.write
        (e.g. ``fmt="%.8f"`` to control numeric precision)

    Examples
    --------
    >>> from petrophys.io import export_las
    >>> export_las(well, "output.las")
    """
    if not HAS_LASIO:
        raise ImportError(
            "lasio is required for LAS file export. "
            "Install with: pip install lasio"
        )

    filepath = Path(filepath)

    # Create LAS file
    las = lasio.LASFile()

    # Set version
    las.version.VERS = lasio.HeaderItem(
        mnemonic="VERS",
        value=version,
        descr="CWLS LOG ASCII STANDARD - VERSION"
    )
    las.version.WRAP = lasio.HeaderItem(
        mnemonic="WRAP",
        value="YES" if wrap else "NO",
        descr="ONE LINE PER DEPTH STEP"
    )

    # Set well header
    header = well.header
    las.well.WELL = lasio.HeaderItem(
        mnemonic="WELL",
        value=header.well_name,
        descr="WELL NAME"
    )
    if header.uwi:
        las.well.UWI = lasio.HeaderItem(
            mnemonic="UWI",
            value=header.uwi,
            descr="UNIQUE WELL ID"
        )
    if header.api:
        las.well.API = lasio.HeaderItem(
            mnemonic="API",
            value=header.api,
            descr="API NUMBER"
        )
    if header.field:
        las.well.FLD = lasio.HeaderItem(
            mnemonic="FLD",
            value=header.field,
            descr="FIELD"
        )
    if header.company:
        las.well.COMP = lasio.HeaderItem(
            mnemonic="COMP",
            value=header.company,
            descr="COMPANY"
        )
    if header.location:
        las.well.LOC = lasio.HeaderItem(
            mnemonic="LOC",
            value=header.location,
            descr="LOCATION"
        )

    # Set depth reference
    if well.top is not None:
        las.well.STRT = lasio.HeaderItem(
            mnemonic="STRT",
            unit=well.depth_unit,
            value=well.top,
            descr="START DEPTH"
        )
    if well.bottom is not None:
        las.well.STOP = lasio.HeaderItem(
            mnemonic="STOP",
            unit=well.depth_unit,
            value=well.bottom,
            descr="STOP DEPTH"
        )
    if well.step is not None:
        las.well.STEP = lasio.HeaderItem(
            mnemonic="STEP",
            unit=well.depth_unit,
            value=well.step,
            descr="STEP"
        )

    # Add null value
    las.well.NULL = lasio.HeaderItem(
        mnemonic="NULL",
        value=-999.25,
        descr="NULL VALUE"
    )

    _export_well_parameters(las, well)
    for item in extra_parameters or []:
        las.params.append(
            lasio.HeaderItem(
                mnemonic=str(item["mnemonic"]),
                unit=str(item.get("unit", "") or ""),
                value=item.get("value", ""),
                descr=str(item.get("descr", "") or ""),
            )
        )
    _export_well_other_comments(las, well)
    if other:
        las.other = "\n".join(part for part in (las.other, other) if part)

    # Add depth curve
    if well.depth is not None:
        las.append_curve(
            "DEPT",
            well.depth,
            unit=well.depth_unit,
            descr="DEPTH"
        )

    # Get curves to export
    if curves is None:
        curves = well.curves
    else:
        curves = [c.upper() for c in curves]

    # Add curves
    for mnemonic in curves:
        if mnemonic not in well._curves:
            continue
        curve = well._curves[mnemonic]

        # Replace non-finite values (NaN, +/-Inf) with the LAS null value.
        # Inf was previously written literally as "inf", breaking re-read round-trips.
        data = curve.data.copy().astype(float)
        data = np.where(np.isfinite(data), data, -999.25)

        las.append_curve(
            mnemonic,
            data,
            unit=curve.unit,
            descr=curve.metadata.description
        )

    # Write file
    las.write(str(filepath), version=version, **kwargs)


def export_csv(
    well: "Well",
    filepath: Union[str, Path],
    curves: Optional[List[str]] = None,
    include_depth: bool = True,
    null_value: str = "",
    **kwargs
):
    """
    Export Well to CSV file.

    Parameters
    ----------
    well : Well
        Well object to export
    filepath : str or Path
        Output file path
    curves : list of str, optional
        Specific curves to export (default: all)
    include_depth : bool, optional
        Whether to include depth column (default: True)
    null_value : str, optional
        String to use for null values (default: empty)
    **kwargs
        Additional arguments passed to DataFrame.to_csv

    Examples
    --------
    >>> from petrophys.io import export_csv
    >>> export_csv(well, "output.csv")
    """
    if curves is None:
        curves = well.curves
    else:
        curves = [c.upper() for c in curves]

    df = well.to_dataframe(curves=curves, include_depth=include_depth)

    # Coerce non-finite values to NaN so Inf is not written literally as "inf"
    # (which would re-read as a float inf rather than a null).
    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle null values
    if null_value != "":
        df = df.fillna(null_value)

    df.to_csv(filepath, index=False, **kwargs)


def export_excel(
    well: "Well",
    filepath: Union[str, Path],
    curves: Optional[List[str]] = None,
    include_depth: bool = True,
    sheet_name: str = "Log Data",
    include_header: bool = True,
    **kwargs
):
    """
    Export Well to Excel file.

    Parameters
    ----------
    well : Well
        Well object to export
    filepath : str or Path
        Output file path (should be .xlsx)
    curves : list of str, optional
        Specific curves to export (default: all)
    include_depth : bool, optional
        Whether to include depth column (default: True)
    sheet_name : str, optional
        Name for the data sheet (default: "Log Data")
    include_header : bool, optional
        Whether to include header information (default: True)
    **kwargs
        Additional arguments passed to DataFrame.to_excel

    Examples
    --------
    >>> from petrophys.io import export_excel
    >>> export_excel(well, "output.xlsx")
    """
    import pandas as pd

    filepath = Path(filepath)
    if filepath.suffix.lower() not in ['.xlsx', '.xls']:
        filepath = filepath.with_suffix('.xlsx')

    if curves is None:
        curves = well.curves
    else:
        curves = [c.upper() for c in curves]

    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Write header sheet if requested
        if include_header:
            header_data = well.header.to_dict()
            header_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in header_data.items()
                if v is not None and v != ""
            ])
            header_df.to_excel(writer, sheet_name="Header", index=False)

        # Write log data
        df = well.to_dataframe(curves=curves, include_depth=include_depth)
        df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)

        # Write curve info
        curve_info = []
        for mnemonic in curves:
            if mnemonic in well._curves:
                curve = well._curves[mnemonic]
                curve_info.append({
                    "Mnemonic": mnemonic,
                    "Unit": curve.unit,
                    "Description": curve.metadata.description,
                    "Min": curve.min(),
                    "Max": curve.max(),
                    "Valid Count": curve.valid_count(),
                    "Null Count": curve.null_count()
                })
        if curve_info:
            curve_df = pd.DataFrame(curve_info)
            curve_df.to_excel(writer, sheet_name="Curve Info", index=False)
