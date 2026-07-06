"""
ASCII and CSV file readers for well log data.

This module provides functionality to read ASCII-formatted well log
data files including CSV, tab-delimited, and space-delimited formats.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd


def read_csv(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: str = "ft",
    well_name: str = "",
    skiprows: int = 0,
    null_values: Optional[List[str]] = None,
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
    from petrophys.base.well import Well

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

    # Create well
    if not well_name:
        well_name = filepath.stem

    well = Well.from_dataframe(
        df,
        depth_column=depth_column.upper(),
        depth_unit=depth_unit,
        name=well_name
    )

    return well


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
    from petrophys.base.well import Well
    from petrophys.base.curve import Curve, CurveMetadata
    from petrophys.base.data_types import DataType

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ASCII file not found: {filepath}")

    # Default null values
    na_values = ["-999", "-999.25", "-999.0", "null", "NULL", "NaN", "nan", ""]
    if null_values:
        na_values.extend(null_values)

    # Read file to detect format if needed
    with open(filepath, 'r') as f:
        lines = f.readlines()

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

    # Create well
    if not well_name:
        well_name = filepath.stem

    well = Well(name=well_name)

    # Set depth
    depth_col = depth_column.upper()
    if depth_col in df.columns:
        well._depth = df[depth_col].values
        well._depth_unit = units.get(depth_col, depth_unit)

    # Add curves
    for col in df.columns:
        if col == depth_col:
            continue

        unit = units.get(col, "")
        metadata = CurveMetadata(
            description=col,
            source="ASCII",
            data_type=DataType.CURVE
        )

        curve = Curve(
            mnemonic=col,
            data=df[col].values,
            unit=unit,
            depth=well._depth,
            depth_unit=well._depth_unit,
            metadata=metadata
        )
        well._curves[col] = curve

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
