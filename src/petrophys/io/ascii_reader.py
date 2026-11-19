"""
ASCII and CSV file readers for well log data.

This module provides functionality to read ASCII-formatted well log
data files including CSV, tab-delimited, and space-delimited formats.
"""

from petrophys._internal.advice import Advice
from typing import TYPE_CHECKING, Callable, Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import csv
import re
import warnings
import numpy as np
import pandas as pd
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.units import normalize_depth_unit

if TYPE_CHECKING:
    from petrophys.model.well import Well



def _require_depth_unit(depth_unit: Optional[str], source: str) -> str:
    """Return the validated depth unit, or explain that one is required."""
    if depth_unit is None or not str(depth_unit).strip():
        raise ValueError(
            Advice(f"{source} does not record its depth unit; pass depth_unit='m' or depth_unit='ft'.", ('depth_unit',))
        )
    return normalize_depth_unit(depth_unit)


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

#: A column header with its unit: "GR (gAPI)" or "DEPTH [m]".
_UNIT_HEADER = re.compile(r"^\s*(.*?\S)\s*(?:\(\s*([^()]*?)\s*\)|\[\s*([^\[\]]*?)\s*\])\s*$")


def _split_unit_header(column: Any) -> Tuple[str, Optional[str]]:
    """The curve name (upper case) and unit of a column header.

    ``"GR (gAPI)"`` and ``"GR [gAPI]"`` give ``("GR", "gAPI")``; a header
    without a bracketed unit gives its name and None.
    """
    text = str(column)
    match = _UNIT_HEADER.match(text)
    if match is None:
        return text.strip().upper(), None
    unit = match.group(2) if match.group(2) is not None else match.group(3)
    return match.group(1).strip().upper(), unit.strip()


def _header_names_and_units(columns: Any, source: str) -> Tuple[List[str], Dict[str, str]]:
    """Curve names of ``columns`` with their header units removed, and the units by name.

    Raises ValueError when two columns become the same curve name.
    """
    names, units, seen = [], {}, {}
    for column in columns:
        name, unit = _split_unit_header(column)
        if name in seen:
            raise ValueError(
                f"{source}: columns {seen[name]!r} and {column!r} both become curve {name!r}; "
                "rename one of them."
            )
        seen[name] = column
        names.append(name)
        if unit:
            units[name] = unit
    return names, units


def _depth_unit_from_header(
    header_unit: Optional[str], depth_unit: Optional[str], depth_column: str, source: str
) -> Optional[str]:
    """The depth unit from ``depth_unit=`` and the depth column header ("DEPTH (m)").

    A header unit stands in for a missing ``depth_unit``; when both are
    given they must agree.
    """
    if not header_unit:
        return depth_unit
    try:
        declared = normalize_depth_unit(header_unit)
    except ValueError:
        return depth_unit  # not a length unit, e.g. "DEPTH (MD)"
    if depth_unit is None or not str(depth_unit).strip():
        return declared
    if normalize_depth_unit(depth_unit, strict=False) != declared:
        raise ValueError(
            f"{source}: the {depth_column} header gives depth unit {header_unit!r}, "
            f"which differs from depth_unit={depth_unit!r}; pass the unit the depths are in."
        )
    return depth_unit


def _set_header_units(well: "Well", units: Dict[str, str]) -> None:
    """Give each curve of ``well`` its unit from the column header."""
    for mnemonic in well.curves:
        if units.get(mnemonic):
            well[mnemonic].unit = units[mnemonic]


def _na_values(null_values: Any = None) -> List[str]:
    """The null values a table is read with: the usual ones and ``null_values`` (a value or list)."""
    values = ["-999", "-999.25", "-999.0", "-9999", "-9999.0", "-9999.25",
              "null", "NULL", "NaN", "nan", ""]
    if null_values is not None:
        if isinstance(null_values, (str, bytes, int, float)):
            null_values = [null_values]
        values.extend(str(value) for value in null_values)
    return values


#: Suffixes of the Excel workbooks :func:`read_excel` reads.
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")

#: A number written with a decimal comma ("2000,5") or a decimal point ("2000.5").
_DECIMAL_COMMA = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")
_DECIMAL_POINT = re.compile(r"^[+-]?\d*\.\d+(?:[eE][+-]?\d+)?$")


def _detect_separator_and_decimal(filepath: Path, skiprows: Any, options: Dict[str, Any]) -> None:
    """Add to ``options`` the separator and decimal mark a delimited text file uses, when not given.

    Without ``sep`` or ``delimiter``, a header line that a semicolon or a
    tab splits into more columns than a comma does gives that separator.
    Without ``decimal`` or ``thousands``, a semicolon- or tab-separated file
    whose first rows write numbers as ``2000,5`` and none as ``2000.5`` is
    read with decimal commas.
    """
    if reproducing_benchmark() or not isinstance(skiprows, int) or options.get("delim_whitespace"):
        return
    lines: List[str] = []
    try:
        with filepath.open("r", encoding=options.get("encoding") or "utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                if index >= skiprows and line.strip():
                    lines.append(line.rstrip("\r\n"))
                if len(lines) > 50:
                    break
    except (OSError, LookupError):
        return
    if not lines:
        return
    separator = options.get("sep", options.get("delimiter"))
    if "sep" not in options and "delimiter" not in options:
        counts = {mark: len(next(csv.reader([lines[0]], delimiter=mark))) for mark in (",", ";", "\t")}
        best = max((";", "\t"), key=lambda mark: counts[mark])
        if counts[best] > counts[","]:
            separator = options["sep"] = best
    if separator in (";", "\t") and "decimal" not in options and "thousands" not in options:
        fields = [field.strip() for line in lines[1:] for field in line.split(separator)]
        if any(_DECIMAL_COMMA.match(field) for field in fields) and not any(
            _DECIMAL_POINT.match(field) for field in fields
        ):
            options["decimal"] = ","


def _read_log_table(
    read: Callable[..., pd.DataFrame], options: Dict[str, Any], source: str, well_column: Optional[str]
) -> pd.DataFrame:
    """Read a log table with ``read(**options)``, its well identifier column as text.

    The identifier column (``well_column``, else WELL, WELLNAME, WELL_NAME,
    UWI or API) is read as text, so a numeric UWI keeps its leading zeros.
    """
    header_options = {
        key: value for key, value in options.items()
        if key not in {"nrows", "dtype", "converters", "usecols", "chunksize", "iterator", "na_values"}
    }
    raw_columns = read(nrows=0, **header_options).columns
    raw_names, _ = _header_names_and_units(raw_columns, source)
    raw_by_upper = dict(zip(raw_names, raw_columns))
    resolved = _resolve_well_column(pd.DataFrame(columns=raw_names), well_column)
    if resolved is not None and not isinstance(options.get("dtype"), (str, type)):
        options["dtype"] = {**(options.get("dtype") or {}), raw_by_upper[resolved]: str}
    return read(**options)


def _well_from_log_table(
    df: pd.DataFrame,
    kind: str,
    filepath: Path,
    depth_column: str,
    depth_unit: Optional[str],
    well_name: str,
    well_column: Optional[str],
) -> "Well":
    """The well of a log table read from a ``kind`` ("CSV" or "Excel") file.

    Column headers ``NAME (unit)`` or ``NAME [unit]`` give curve ``NAME``
    with that unit, the depth column header's unit stands in for a missing
    ``depth_unit``, and the well identifier column names the well.
    """
    from petrophys.model.well import Well

    source = f"{kind} file {filepath.name!r}"
    # Clean column names; "GR (gAPI)" and "GR [gAPI]" give curve GR in gAPI.
    df.columns, units = _header_names_and_units(df.columns, source)
    depth_column = _split_unit_header(depth_column)[0]
    if depth_column not in df.columns:
        hint = ""
        if len(df.columns) == 1 and any(sep in str(df.columns[0]) for sep in (";", "\t", "|", " ")):
            hint = f" The header was read as the single column {str(df.columns[0])!r}; " + (
                "pass the separator the file uses (e.g. sep=';')." if kind == "CSV"
                else "split the text into columns in Excel, or save the sheet as CSV."
            )
        raise ValueError(
            f"Depth column {depth_column!r} not found. Available columns: "
            f"{list(df.columns)}. Set depth_column explicitly.{hint}"
        )

    candidate_well_column = _resolve_well_column(df, well_column)
    if candidate_well_column is not None:
        identifiers = _well_identifiers(df[candidate_well_column])
        if len(identifiers) > 1:
            raise ValueError(
                f"{kind} contains {len(identifiers)} wells in {candidate_well_column!r}; " + (
                    "use read_csv_wells() to keep them separate." if kind == "CSV"
                    else "save the sheet as CSV and use read_csv_wells() to keep them separate."
                )
            )
        if not well_name and identifiers:
            well_name = identifiers[0]

    if not well_name:
        well_name = filepath.stem

    _warn_on_coerced_values(
        df, [c for c in (candidate_well_column,) if c], source
    )
    if candidate_well_column is not None and candidate_well_column != depth_column:
        df = df.drop(columns=[candidate_well_column])
    depth_unit = _depth_unit_from_header(units.get(depth_column), depth_unit, depth_column, source)
    well = Well.from_dataframe(
        df,
        depth_column=depth_column.upper(),
        depth_unit=_require_depth_unit(depth_unit, source),
        name=well_name,
        header_units=False,
    )
    _set_header_units(well, units)
    if candidate_well_column in ("UWI", "API") and len(identifiers) == 1:
        setattr(well.header, candidate_well_column.lower(), identifiers[0])

    return well


def _excel_sheet(workbook: pd.ExcelFile, sheet_name: Any, depth_column: str, skiprows: int, source: str) -> Any:
    """The sheet of ``workbook`` that holds the log table.

    ``sheet_name`` (a name or position) when given, else the only sheet,
    else the first sheet whose header row has ``depth_column`` (``Log Data``
    in a workbook :func:`~petrophys.io.export_excel` wrote).
    """
    if sheet_name is not None:
        return sheet_name
    names = workbook.sheet_names
    if len(names) == 1:
        return names[0]
    wanted = _split_unit_header(depth_column)[0]
    for name in names:
        columns = pd.read_excel(workbook, sheet_name=name, skiprows=skiprows, nrows=0).columns
        if wanted in {_split_unit_header(column)[0] for column in columns}:
            return name
    raise ValueError(
        f"{source}: no sheet has the depth column {wanted!r} (sheets: {names}); "
        "pass sheet_name and depth_column."
    )


def _depth_header_unit(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    skiprows: int = 0,
    sheet_name: Any = None,
    **kwargs: Any
) -> Optional[str]:
    """The depth unit ("m" or "ft") the depth column header of a log table states, or None.

    ``filepath`` is a CSV, TSV or Excel file read with the options of
    :func:`read_csv` or :func:`read_excel`; ``DEPTH (m)`` gives "m".
    """
    filepath = Path(filepath)
    wanted = _split_unit_header(depth_column)[0]
    try:
        if filepath.suffix.lower() in EXCEL_SUFFIXES:
            with pd.ExcelFile(filepath) as workbook:
                sheet = _excel_sheet(workbook, sheet_name, depth_column, skiprows, filepath.name)
                columns = pd.read_excel(workbook, sheet_name=sheet, skiprows=skiprows, nrows=0).columns
        else:
            options = {key: kwargs[key] for key in ("sep", "delimiter", "encoding") if key in kwargs}
            _detect_separator_and_decimal(filepath, skiprows, options)
            columns = pd.read_csv(filepath, skiprows=skiprows, nrows=0, **options).columns
    except (ValueError, OSError, UnicodeDecodeError, ImportError):
        return None
    for column in columns:
        name, unit = _split_unit_header(column)
        if name == wanted and unit:
            try:
                return normalize_depth_unit(unit)
            except ValueError:
                return None
    return None


def read_csv(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: Optional[str] = None,
    well_name: str = "",
    skiprows: int = 0,
    null_values: Any = None,
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
        Name of depth column (default: "DEPTH"); a header with a unit,
        ``DEPTH (m)`` or ``DEPTH [m]``, is found by its name
    depth_unit : {"m", "ft"}
        Unit of the depth column. Required unless the depth column header
        gives it (``DEPTH (m)``); when both are given they must agree.
    well_name : str, optional
        Name for the well
    skiprows : int, optional
        Number of rows to skip at start
    null_values : scalar or list, optional
        Further null values (e.g. ``-99999``) besides -999, -999.25, -9999,
        -9999.25, "null" and "NaN"
    well_column : str, optional
        Column identifying the well (default: the first of WELL, WELLNAME,
        WELL_NAME, UWI and API present). It is read as text, so identifiers
        keep their leading zeros, names the well and is not a curve; a UWI
        or API column also sets ``header.uwi`` or ``header.api``.
    **kwargs
        Additional arguments passed to pd.read_csv (e.g. ``sep=";"``,
        ``decimal=","`` or ``thousands=","``). Without ``sep``, a header
        line that a semicolon or a tab splits into more columns than a comma
        does gives the separator; without ``decimal`` and ``thousands``, a
        semicolon- or tab-separated file whose first rows write numbers as
        ``2000,5`` and none as ``2000.5`` is read with decimal commas.

    Returns
    -------
    Well
        Well object with loaded data. A column header ``NAME (unit)`` or
        ``NAME [unit]``, as :func:`~petrophys.io.export_csv` writes, gives
        curve ``NAME`` with that unit.

    Raises
    ------
    ValueError
        If the depth column is missing or has no numeric value, two
        columns become the same curve name (they differ only in case or
        unit), or the depth header's unit differs from ``depth_unit``.

    Examples
    --------
    >>> from petrophys.io import read_csv
    >>> well = read_csv("well_data.csv", depth_column="MD", depth_unit="m")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    _detect_separator_and_decimal(filepath, skiprows, kwargs)

    def read(**options: Any) -> pd.DataFrame:
        return pd.read_csv(filepath, skiprows=skiprows, **options)

    df = _read_log_table(
        read, {"na_values": _na_values(null_values), **kwargs}, f"CSV file {filepath.name!r}", well_column
    )
    return _well_from_log_table(df, "CSV", filepath, depth_column, depth_unit, well_name, well_column)


def read_excel(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: Optional[str] = None,
    well_name: str = "",
    sheet_name: Any = None,
    skiprows: int = 0,
    null_values: Any = None,
    well_column: Optional[str] = None,
    **kwargs
) -> "Well":
    """
    Read a log table from an Excel workbook and return Well object.

    The sheet is read as :func:`read_csv` reads a CSV file with the same
    table: one row per depth, a depth column and one column per curve.

    Parameters
    ----------
    filepath : str or Path
        Path to the workbook (``.xlsx``, ``.xlsm`` or ``.xls``)
    depth_column : str, optional
        Name of depth column (default: "DEPTH"); a header with a unit,
        ``DEPTH (m)`` or ``DEPTH [m]``, is found by its name
    depth_unit : {"m", "ft"}
        Unit of the depth column. Required unless the depth column header
        gives it (``DEPTH (m)``); when both are given they must agree.
    well_name : str, optional
        Name for the well (default: the well identifier column's value,
        else the file name without its suffix)
    sheet_name : str or int, optional
        Sheet name or position (0 is the first). Default: the only sheet,
        or the first sheet whose header row has the depth column.
    skiprows : int, optional
        Number of rows to skip above the header row
    null_values : scalar or list, optional
        Further null values (e.g. ``-99999``) besides -999, -999.25, -9999,
        -9999.25, "null", "NaN" and empty cells
    well_column : str, optional
        Column identifying the well (default: the first of WELL, WELLNAME,
        WELL_NAME, UWI and API present). It is read as text, names the well
        and is not a curve; a UWI or API column also sets ``header.uwi`` or
        ``header.api``.
    **kwargs
        Additional arguments passed to pd.read_excel

    Returns
    -------
    Well
        Well object with loaded data. A column header ``NAME (unit)`` or
        ``NAME [unit]`` gives curve ``NAME`` with that unit.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.
    ValueError
        If no sheet has the depth column, the depth column has no numeric
        value, two columns become the same curve name (they differ only in
        case or unit), the sheet holds several wells, or the depth header's
        unit differs from ``depth_unit``.

    Examples
    --------
    >>> from petrophys.io import read_excel
    >>> well = read_excel("well_logs.xlsx", depth_unit="m")  # doctest: +SKIP
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    source = f"Excel file {filepath.name!r}"
    with pd.ExcelFile(filepath) as workbook:
        sheet = _excel_sheet(workbook, sheet_name, depth_column, skiprows, source)

        def read(**options: Any) -> pd.DataFrame:
            return pd.read_excel(workbook, sheet_name=sheet, skiprows=skiprows, **options)

        df = _read_log_table(read, {"na_values": _na_values(null_values), **kwargs}, source, well_column)
    return _well_from_log_table(df, "Excel", filepath, depth_column, depth_unit, well_name, well_column)


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
    null_values: Any = None,
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

    na_values = _na_values(null_values)
    _detect_separator_and_decimal(path, skiprows, kwargs)

    # Read the identifier column as text in every chunk, so a numeric UWI is
    # not parsed as int in one chunk and float in another ("1001" vs
    # "1001.0") and keeps its leading zeros.
    header_options = {
        key: value for key, value in kwargs.items()
        if key not in {"nrows", "dtype", "converters", "usecols"}
    }
    raw_columns = pd.read_csv(path, skiprows=skiprows, nrows=0, **header_options).columns
    source = f"CSV file {path.name!r}"
    raw_names, units = _header_names_and_units(raw_columns, source)
    raw_by_upper = dict(zip(raw_names, raw_columns))
    resolved = _resolve_well_column(pd.DataFrame(columns=raw_names), well_column)
    depth_column = _split_unit_header(depth_column)[0]
    depth_unit = _depth_unit_from_header(units.get(depth_column), depth_unit, depth_column, source)
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
        chunk.columns = [_split_unit_header(column)[0] for column in chunk.columns]
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
            header_units=False,
        )
        _set_header_units(wells[name], units)
    return wells


def read_ascii(
    filepath: Union[str, Path],
    depth_column: str = "DEPTH",
    depth_unit: Optional[str] = None,
    well_name: str = "",
    delimiter: Optional[str] = None,
    header_line: int = 0,
    skiprows: int = 0,
    null_values: Any = None,
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
        Unit of the depth column. Required unless the units line
        (``units_line``) or the depth column header (``DEPTH (m)``,
        ``DEPTH [m]``) gives it; the units line takes precedence, and a
        header unit must agree with ``depth_unit``.
    well_name : str, optional
        Name for the well
    delimiter : str, optional
        Column delimiter (default: auto-detect tab, semicolon, comma or
        whitespace from the first data row)
    header_line : int, optional
        Line number containing column headers (0-indexed)
    skiprows : int, optional
        Number of rows to skip before header
    null_values : scalar or list, optional
        Further null values (e.g. ``-99999``) besides -999, -999.25, -9999,
        -9999.25, "null" and "NaN"
    column_names : list of str, optional
        Column names if not in file
    units_line : int, optional
        Line number containing units. A header ``NAME (unit)`` or
        ``NAME [unit]`` gives curve ``NAME`` with that unit; a unit on the
        units line takes precedence.
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
    na_values = _na_values(null_values)

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
    source = f"ASCII file {Path(filepath).name!r}"
    # "GR (gAPI)" and "GR [gAPI]" give curve GR in gAPI.
    df.columns, header_units = _header_names_and_units(df.columns, source)

    # Get units if available; a units line takes precedence over header units.
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

    depth_col = _split_unit_header(depth_column)[0]
    requested_name = well_name
    file_depth_unit = units.get(depth_col)
    if not file_depth_unit:
        depth_unit = _depth_unit_from_header(header_units.get(depth_col), depth_unit, depth_col, source)
    units = {**header_units, **{name: unit for name, unit in units.items() if unit}}
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
        header_units=False,
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
