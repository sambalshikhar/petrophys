"""
ASCII and CSV file readers for well log data.

This module provides functionality to read ASCII-formatted well log
data files including CSV, tab-delimited, and space-delimited formats.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def read_csv(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: str = "ft",
    well_name: str = "",
    skiprows: int = 0,
    null_values: Optional[List[str]] = None,
    well_column: Optional[str] = None,
    **kwargs
) -> "Well":
    """
    Read CSV file and return Well object.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    depth_column : str, optional
        Name of depth column (default: "DEPTH")
    depth_unit : str, optional
        Unit for depth values (default: "ft")
    well_name : str, optional
        Name for the well
    skiprows : int, optional
        Number of rows to skip at start
    null_values : list of str, optional
        Additional strings to recognize as null
    **kwargs
        Additional arguments passed to pd.read_csv

    Returns
    -------
    Well
        Well object with loaded data

    Examples
    --------
    >>> from petrophys.io import read_csv
    >>> well = read_csv("well_data.csv", depth_column="MD")
    """
    from petrophys.model.well import Well

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Default null values
    na_values = ["-999", "-999.25", "-999.0", "null", "NULL", "NaN", "nan", ""]
    if null_values:
        na_values.extend(null_values)

    # Read CSV
    df = pd.read_csv(
        filepath,
        skiprows=skiprows,
        na_values=na_values,
        **kwargs
    )

    # Clean column names
    df.columns = [c.strip().upper() for c in df.columns]

    candidate_well_column = _resolve_well_column(df, well_column)
    if candidate_well_column is not None:
        identifiers = _well_identifiers(df[candidate_well_column])
        if len(identifiers) > 1:
            raise ValueError(
                f"CSV contains {len(identifiers)} wells in "
                f"{candidate_well_column!r}; use read_csv_wells() to keep them separate."
            )
        if not well_name and identifiers:
            well_name = identifiers[0]

    if not well_name:
        well_name = filepath.stem

    well = Well.from_dataframe(
        df,
        depth_column=depth_column.upper(),
        depth_unit=depth_unit,
        name=well_name
    )

    return well


def _resolve_well_column(
    frame: pd.DataFrame,
    requested: Optional[str],
) -> Optional[str]:
    if requested is not None:
        column = str(requested).strip().upper()
        if column not in frame.columns:
            raise KeyError(f"Well identifier column not found: {requested}")
        return column
    return next(
        (
            candidate
            for candidate in ("WELL", "WELLNAME", "WELL_NAME", "UWI", "API")
            if candidate in frame.columns
        ),
        None,
    )


def _well_identifiers(series: pd.Series) -> List[str]:
    values = series.dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def read_csv_wells(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: str = "ft",
    well_column: Optional[str] = None,
    skiprows: int = 0,
    null_values: Optional[List[str]] = None,
    chunksize: int = 100_000,
    **kwargs,
) -> Dict[str, "Well"]:
    """Read a long-form multi-well CSV without mixing well identities."""
    from petrophys.model.well import Well

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    na_values = ["-999", "-999.25", "-999.0", "null", "NULL", "NaN", "nan", ""]
    if null_values:
        na_values.extend(null_values)

    grouped: Dict[str, List[pd.DataFrame]] = {}
    identifier_column: Optional[str] = None
    for chunk in pd.read_csv(
        path,
        skiprows=skiprows,
        na_values=na_values,
        chunksize=chunksize,
        **kwargs,
    ):
        chunk.columns = [str(column).strip().upper() for column in chunk.columns]
        if identifier_column is None:
            identifier_column = _resolve_well_column(chunk, well_column)
            if identifier_column is None:
                raise ValueError(
                    "No well identifier column found; pass well_column explicitly"
                )
        for identifier, rows in chunk.groupby(identifier_column, dropna=True, sort=False):
            name = str(identifier).strip()
            if name:
                grouped.setdefault(name, []).append(rows)

    wells: Dict[str, Well] = {}
    for name, frames in grouped.items():
        frame = pd.concat(frames, axis=0, ignore_index=True)
        wells[name] = Well.from_dataframe(
            frame,
            depth_column=depth_column.upper(),
            depth_unit=depth_unit,
            name=name,
        )
    return wells


def read_ascii(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: str = "ft",
    well_name: str = "",
    delimiter: Optional[str] = None,
    header_line: int = 0,
    skiprows: int = 0,
    null_values: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    units_line: Optional[int] = None,
    **kwargs
) -> "Well":
    """
    Read ASCII well log file and return Well object.

    This function handles various ASCII formats including space-delimited,
    tab-delimited, and custom delimiters.

    Parameters
    ----------
    filepath : str or Path
        Path to ASCII file
    depth_column : str, optional
        Name of depth column (default: "DEPTH")
    depth_unit : str, optional
        Unit for depth values (default: "ft")
    well_name : str, optional
        Name for the well
    delimiter : str, optional
        Column delimiter (default: auto-detect)
    header_line : int, optional
        Line number containing column headers (0-indexed)
    skiprows : int, optional
        Number of rows to skip before header
    null_values : list of str, optional
        Strings to recognize as null values
    column_names : list of str, optional
        Column names if not in file
    units_line : int, optional
        Line number containing units

    Returns
    -------
    Well
        Well object with loaded data
    """
    from petrophys.model.well import Well

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ASCII file not found: {filepath}")

    # Default null values
    na_values = ["-999", "-999.25", "-999.0", "null", "NULL", "NaN", "nan", ""]
    if null_values:
        na_values.extend(null_values)

    # Probe only the lines needed for delimiter/header/unit detection. Pandas
    # performs the single full-file read below.
    probe_end = max(header_line + skiprows + 1, units_line or 0) + 1
    lines = []
    with filepath.open("r", encoding=kwargs.get("encoding", "utf-8"), errors="replace") as stream:
        for index, line in enumerate(stream):
            lines.append(line)
            if index >= probe_end:
                break
    if not lines:
        raise ValueError(f"ASCII file is empty: {filepath}")

    # Auto-detect delimiter if not specified
    if delimiter is None:
        sample_line = lines[header_line + skiprows + 1] if len(lines) > header_line + skiprows + 1 else lines[-1]
        if '\t' in sample_line:
            delimiter = '\t'
        elif ',' in sample_line:
            delimiter = ','
        else:
            delimiter = r'\s+'  # whitespace

    if column_names:
        # column_names means the file has no header row: only skip rows the
        # caller explicitly asked for (plus a units line, if any).
        skiprow_indices = set(range(skiprows))
        if units_line is not None:
            skiprow_indices.add(units_line)
        read_skiprows = sorted(skiprow_indices) if skiprow_indices else None
    else:
        skiprow_indices = set(range(skiprows))
        if units_line is not None and units_line != header_line:
            skiprow_indices.add(units_line)
        read_skiprows = sorted(skiprow_indices) if skiprow_indices else None

    # Read data
    if column_names:
        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            names=column_names,
            skiprows=read_skiprows,
            na_values=na_values,
            **kwargs
        )
    else:
        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            skiprows=read_skiprows,
            header=header_line,
            na_values=na_values,
            **kwargs
        )

    # Clean column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Get units if available
    units = {}
    if units_line is not None and units_line < len(lines):
        unit_parts = lines[units_line].strip().split(delimiter if delimiter != r'\s+' else None)
        if len(unit_parts) == len(df.columns):
            units = dict(zip(df.columns, unit_parts))

    depth_col = depth_column.upper()
    requested_name = well_name
    well = Well.from_dataframe(
        df,
        depth_column=depth_col,
        depth_unit=units.get(depth_col, depth_unit),
        name=requested_name,
    )
    if not well.name:
        well.name = filepath.stem

    for mnemonic in well.curves:
        curve = well[mnemonic]
        curve.unit = units.get(mnemonic, "")
        curve.metadata.description = mnemonic
        curve.metadata.source = "ASCII"

    skipped = well.get_parameter("SKIPPED_NON_NUMERIC_COLUMNS", [])
    if skipped:
        warnings.warn(
            "Skipped non-numeric ASCII metadata columns: " + ", ".join(skipped),
            stacklevel=2,
        )

    return well


def detect_file_format(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect the format of an ASCII well log file.

    Parameters
    ----------
    filepath : str or Path
        Path to file

    Returns
    -------
    dict
        Dictionary with detected format parameters
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = [f.readline() for _ in range(20)]

    # Check for common patterns
    format_info = {
        "delimiter": None,
        "header_line": 0,
        "skiprows": 0,
        "has_units": False,
    }

    # Count potential delimiters in data lines
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line or line.startswith('#'):
            format_info["skiprows"] = i
            continue

        # Check delimiter
        if '\t' in line:
            format_info["delimiter"] = '\t'
        elif ',' in line:
            format_info["delimiter"] = ','
        else:
            format_info["delimiter"] = r'\s+'
        break

    return format_info
