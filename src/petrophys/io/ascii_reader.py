"""
ASCII and CSV file readers for well log data.

This module provides functionality to read ASCII-formatted well log
data files including CSV, tab-delimited, and space-delimited formats.
"""

from typing import TYPE_CHECKING, Optional, Union, List, Dict, Any
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from petrophys.model.units import normalize_depth_unit

if TYPE_CHECKING:
    from petrophys.model.well import Well



def _require_depth_unit(depth_unit: Optional[str], source: str) -> str:
    """Return the validated depth unit, or explain that one is required."""
    if depth_unit is None or not str(depth_unit).strip():
        raise ValueError(
            f"{source} does not record its depth unit; pass depth_unit='m' or depth_unit='ft'."
        )
    return normalize_depth_unit(depth_unit)


def _check_case_duplicates(columns: Any, source: str) -> None:
    """Raise ValueError when two columns become the same (upper-case) curve name."""
    seen: Dict[str, Any] = {}
    for column in columns:
        key = str(column).strip().upper()
        if key in seen:
            raise ValueError(
                f"{source}: columns {seen[key]!r} and {column!r} both become curve {key!r}; "
                "rename one of them."
            )
        seen[key] = column


def _warn_on_coerced_values(df: pd.DataFrame, skip: List[str], source: str) -> None:
    """Warn when text in mostly numeric columns is read as missing (NaN)."""
    coerced = {}
    for column in df.columns:
        if column in skip or pd.api.types.is_numeric_dtype(df[column]):
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if not numeric.notna().any():
            continue  # a text column; Well.from_dataframe reports it separately
        lost = series.notna() & numeric.isna()
        if lost.any():
            coerced[column] = series[lost].astype(str).unique()[:5].tolist()
    if coerced:
        details = "; ".join(f"{column}: {values}" for column, values in coerced.items())
        warnings.warn(
            f"{source}: non-numeric values in numeric columns are read as missing (NaN) - {details}",
            UserWarning,
            stacklevel=3,
        )

def read_csv(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: Optional[str] = None,
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
    depth_unit : {"m", "ft"}
        Unit of the depth column. Required: CSV files do not record it.
    well_name : str, optional
        Name for the well
    skiprows : int, optional
        Number of rows to skip at start
    null_values : list of str, optional
        Additional strings to recognize as null
    well_column : str, optional
        Column identifying the well (default: the first of WELL, WELLNAME,
        WELL_NAME, UWI and API present). It is read as text, so identifiers
        keep their leading zeros, names the well and is not a curve; a UWI
        or API column also sets ``header.uwi`` or ``header.api``.
    **kwargs
        Additional arguments passed to pd.read_csv (e.g. ``thousands=","``
        or ``decimal=","``)

    Returns
    -------
    Well
        Well object with loaded data

    Raises
    ------
    ValueError
        If the depth column is missing or has no numeric value, or two
        columns differ only in case.

    Examples
    --------
    >>> from petrophys.io import read_csv
    >>> well = read_csv("well_data.csv", depth_column="MD", depth_unit="m")
    """
    from petrophys.model.well import Well

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Default null values
    na_values = ["-999", "-999.25", "-999.0", "null", "NULL", "NaN", "nan", ""]
    if null_values:
        na_values.extend(null_values)

    # Read the well identifier column as text, so a numeric UWI keeps its
    # leading zeros.
    header_options = {
        key: value for key, value in kwargs.items()
        if key not in {"nrows", "dtype", "converters", "usecols", "chunksize", "iterator"}
    }
    raw_columns = pd.read_csv(filepath, skiprows=skiprows, nrows=0, **header_options).columns
    source = f"CSV file {filepath.name!r}"
    _check_case_duplicates(raw_columns, source)
    raw_by_upper = {str(column).strip().upper(): column for column in raw_columns}
    resolved = _resolve_well_column(pd.DataFrame(columns=list(raw_by_upper)), well_column)
    if resolved is not None and not isinstance(kwargs.get("dtype"), (str, type)):
        kwargs["dtype"] = {**(kwargs.get("dtype") or {}), raw_by_upper[resolved]: str}

    # Read CSV
    df = pd.read_csv(
        filepath,
        skiprows=skiprows,
        na_values=na_values,
        **kwargs
    )

    # Clean column names
    df.columns = [c.strip().upper() for c in df.columns]
    depth_column = depth_column.strip().upper()
    if depth_column not in df.columns:
        raise ValueError(
            f"Depth column {depth_column!r} not found. Available columns: "
            f"{list(df.columns)}. Set depth_column explicitly."
        )

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

    _warn_on_coerced_values(
        df, [c for c in (candidate_well_column,) if c], f"CSV file {filepath.name!r}"
    )
    if candidate_well_column is not None and candidate_well_column != depth_column:
        df = df.drop(columns=[candidate_well_column])
    well = Well.from_dataframe(
        df,
        depth_column=depth_column.upper(),
        depth_unit=_require_depth_unit(depth_unit, f"CSV file {filepath.name!r}"),
        name=well_name
    )
    if candidate_well_column in ("UWI", "API") and len(identifiers) == 1:
        setattr(well.header, candidate_well_column.lower(), identifiers[0])

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
    depth_unit: Optional[str] = None,
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

    # Read the identifier column as text in every chunk, so a numeric UWI is
    # not parsed as int in one chunk and float in another ("1001" vs
    # "1001.0") and keeps its leading zeros.
    header_options = {
        key: value for key, value in kwargs.items()
        if key not in {"nrows", "dtype", "converters", "usecols"}
    }
    raw_columns = pd.read_csv(path, skiprows=skiprows, nrows=0, **header_options).columns
    _check_case_duplicates(raw_columns, f"CSV file {path.name!r}")
    raw_by_upper = {str(column).strip().upper(): column for column in raw_columns}
    resolved = _resolve_well_column(pd.DataFrame(columns=list(raw_by_upper)), well_column)
    if resolved is not None and not isinstance(kwargs.get("dtype"), (str, type)):
        kwargs["dtype"] = {**(kwargs.get("dtype") or {}), raw_by_upper[resolved]: str}

    grouped: Dict[str, List[pd.DataFrame]] = {}
    identifier_column: Optional[str] = None
    blank_identifier_rows = 0
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
        blank_identifier_rows += int(
            (chunk[identifier_column].isna()
             | (chunk[identifier_column].astype(str).str.strip() == "")).sum()
        )
        for identifier, rows in chunk.groupby(identifier_column, dropna=True, sort=False):
            name = str(identifier).strip()
            if name:
                grouped.setdefault(name, []).append(rows)

    if blank_identifier_rows:
        warnings.warn(
            f"CSV file {path.name!r}: {blank_identifier_rows} row(s) with a blank "
            f"{identifier_column!r} were not assigned to any well and are not read.",
            UserWarning,
            stacklevel=2,
        )

    wells: Dict[str, Well] = {}
    for name, frames in grouped.items():
        frame = pd.concat(frames, axis=0, ignore_index=True)
        _warn_on_coerced_values(frame, [identifier_column], f"CSV file {path.name!r}, well {name!r}")
        wells[name] = Well.from_dataframe(
            frame,
            depth_column=depth_column.upper(),
            depth_unit=_require_depth_unit(depth_unit, f"CSV file {Path(filepath).name!r}"),
            name=name,
        )
    return wells


def read_ascii(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: Optional[str] = None,
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
    depth_unit : {"m", "ft"}
        Unit of the depth column. Required: CSV files do not record it.
    well_name : str, optional
        Name for the well
    delimiter : str, optional
        Column delimiter (default: auto-detect tab, semicolon, comma or
        whitespace from the first data row)
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
    **kwargs
        Additional arguments passed to pd.read_csv (e.g. ``decimal=","``
        or ``thousands=","``)

    Returns
    -------
    Well
        Well object with loaded data

    Raises
    ------
    ValueError
        If the depth column is missing or has no numeric value, or two
        columns differ only in case.
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
        header_text = lines[header_line + skiprows] if len(lines) > header_line + skiprows else ""
        if '\t' in sample_line:
            delimiter = '\t'
        elif ';' in sample_line:
            delimiter = ';'
        elif ',' in sample_line and (column_names or ',' in header_text):
            delimiter = ','
        else:
            # Whitespace, also for rows with decimal commas ("2000,5 45,3")
            # under a header without commas.
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
    _check_case_duplicates(df.columns, f"ASCII file {filepath.name!r}")
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Get units if available
    units = {}
    if units_line is not None and units_line < len(lines):
        unit_parts = lines[units_line].strip().split(delimiter if delimiter != r'\s+' else None)
        if len(unit_parts) == len(df.columns):
            units = dict(zip(df.columns, unit_parts))
        else:
            warnings.warn(
                f"ASCII file {filepath.name!r}: units line {units_line} has "
                f"{len(unit_parts)} entries for {len(df.columns)} columns; units are ignored.",
                UserWarning,
                stacklevel=2,
            )

    depth_col = depth_column.upper()
    requested_name = well_name
    source = f"ASCII file {Path(filepath).name!r}"
    file_depth_unit = units.get(depth_col)
    if file_depth_unit and depth_unit is not None and str(depth_unit).strip():
        file_unit = normalize_depth_unit(file_depth_unit, strict=False)
        if file_unit != normalize_depth_unit(depth_unit, strict=False):
            warnings.warn(
                f"{source}: the units line gives depth unit {file_depth_unit!r}, which "
                f"differs from depth_unit={depth_unit!r}; the file's unit is used.",
                UserWarning,
                stacklevel=2,
            )
    _warn_on_coerced_values(df, [depth_col], source)
    well = Well.from_dataframe(
        df,
        depth_column=depth_col,
        depth_unit=_require_depth_unit(file_depth_unit or depth_unit, source),
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
        lines = [line for line in (f.readline() for _ in range(20)) if line]

    # Check for common patterns
    format_info = {
        "delimiter": None,
        "header_line": 0,
        "skiprows": 0,
        "has_units": False,
    }

    # Leading blank or comment lines come before the header; skiprows counts
    # them and header_line is relative to the remaining lines.
    content = [
        (i, line.strip()) for i, line in enumerate(lines)
        if line.strip() and not line.strip().startswith('#')
    ]
    if not content:
        return format_info
    format_info["skiprows"] = content[0][0]

    # The delimiter is taken from the first line after the header, or from
    # the header itself in a header-only file.
    sample = content[1][1] if len(content) > 1 else content[0][1]
    if '\t' in sample:
        format_info["delimiter"] = '\t'
    elif ',' in sample:
        format_info["delimiter"] = ','
    else:
        format_info["delimiter"] = r'\s+'

    # A units line follows the header and holds no numbers.
    if len(content) > 2:
        split_on = None if format_info["delimiter"] == r'\s+' else format_info["delimiter"]
        tokens = [token.strip() for token in content[1][1].split(split_on)]
        format_info["has_units"] = not any(
            pd.notna(pd.to_numeric(token, errors="coerce")) for token in tokens if token
        )

    return format_info
