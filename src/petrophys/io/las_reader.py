"""
LAS file reader for well log data.

This module provides functionality to read LAS (Log ASCII Standard)
files and convert them to Well objects.
"""

from typing import TYPE_CHECKING, Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import re
import warnings
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.units import normalize_depth_unit

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


#: Null values common in LAS files. Data values equal to one of them are read
#: as missing when the header NULL is another value or is absent.
COMMON_LAS_NULLS = (-999.25, -999.0, -9999.0)

#: ~Well items the reader maps to WellHeader fields (and export_las writes
#: from them); other ~Well items are kept in ``WellHeader.las_well_items``.
_MAPPED_WELL_ITEMS = frozenset({
    "STRT", "STOP", "STEP", "NULL", "WELL", "UWI", "API", "FLD", "COMP",
    "LOC", "CNTY", "STAT", "CTRY", "TD", "BS", "BHT",
})

#: Bit size units and their factor to inches.
_INCHES_PER_UNIT = {
    "in": 1.0, "ins": 1.0, "inch": 1.0, "inches": 1.0, '"': 1.0,
    "mm": 1.0 / 25.4, "cm": 1.0 / 2.54, "m": 1.0 / 0.0254,
}

_FORTRAN_EXPONENT = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))[dD]([+-]?\d+)$")
_LEADING_ZERO_NUMBER = re.compile(r"^0\d+$")
_HEADER_LINE = re.compile(r"^([^.]*)\.(\S*)\s(.*)$")
_SCANNED_DATA_ROWS = 20


def _parse_token(token: Any) -> Optional[float]:
    """The number a ~A token holds (Fortran ``D`` exponents included), else None."""
    text = str(token).strip()
    try:
        return float(text)
    except ValueError:
        match = _FORTRAN_EXPONENT.match(text)
        if match:
            return float(f"{match.group(1)}e{match.group(2)}")
    return None


def _numeric_values(data: Any, null: Optional[float]) -> Tuple[Optional[np.ndarray], List[Tuple[int, str]]]:
    """Float values of a ~A column and the ``(row, token)`` pairs that are not numbers.

    lasio leaves a column as text when one of its tokens is not a number.
    Such a column is converted token by token: Fortran ``D`` exponents
    (``1.2345D+02``) are read, the header NULL becomes NaN and other
    unreadable tokens (``-1.#IND``, ``1.#QNAN``, ``****``, ``--``) become NaN.
    A column without any number gives None.
    """
    array = np.asarray(data)
    if array.dtype.kind in "fiub":
        return array.astype(float), []
    values = np.full(array.shape, np.nan)
    bad: List[Tuple[int, str]] = []
    readable = 0
    for row, token in enumerate(array.tolist()):
        if token is None or not str(token).strip():
            continue
        number = _parse_token(token)
        if number is None:
            bad.append((row, str(token).strip()))
            continue
        values[row] = number
        readable += 1
    if bad and not readable:
        return None, bad
    if null is not None:
        values[values == null] = np.nan
    return values, bad


def _is_data_section(title: str) -> bool:
    """True for a ~A (LAS 1.2/2.0) or ``~..._Data`` (LAS 3.0) section title."""
    text = title.strip().upper()
    return text.startswith("~A") or (text.startswith("~") and "_DATA" in text.split("|")[0])


def _scan_las_text(path: Path, encoding: str) -> Dict[str, Any]:
    """The header lines, section titles and first ~A rows of a LAS file."""
    header: List[str] = []
    data: List[str] = []
    sections: List[str] = []
    in_data = False
    with open(path, "r", encoding=encoding, errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("~"):
                if in_data:
                    break
                sections.append(stripped.split()[0] if stripped.split() else stripped)
                in_data = _is_data_section(stripped)
                if not in_data:
                    header.append(line)
                continue
            if not in_data:
                header.append(line)
            elif stripped and not stripped.startswith("#"):
                data.append(stripped)
                if len(data) >= _SCANNED_DATA_ROWS:
                    break
    return {
        "header": header,
        "data": data,
        "sections": sections,
        "has_data_section": any(_is_data_section(title) for title in sections),
        "lowercase_sections": any(len(title) > 1 and title[1].islower() for title in sections),
    }


def _raw_header_values(lines: List[str]) -> Dict[Tuple[str, str], List[str]]:
    """Header values as written, by (section letter, mnemonic), in file order."""
    values: Dict[Tuple[str, str], List[str]] = {}
    section = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("~"):
            section = stripped[1:2].upper()
            continue
        match = _HEADER_LINE.match(stripped)
        if section not in ("W", "P") or match is None:
            continue
        value = match.group(3)
        if ":" in value:
            value = value.rpartition(":")[0]
        values.setdefault((section, match.group(1).strip().upper()), []).append(value.strip())
    return values


def _lasio_read(source: Any, name: str, **kwargs):
    """``lasio.read`` with readable errors."""
    try:
        return lasio.read(source, **kwargs)
    except (KeyError, IndexError, ValueError, lasio.exceptions.LASHeaderError,
            lasio.exceptions.LASDataError) as exc:
        detail = str(exc).strip("'\"") or type(exc).__name__
        raise ValueError(f"{name} is not a readable LAS file: {detail}") from exc


def _comma_delimited(las, sample_rows: List[str]) -> bool:
    """True when the ~A rows are comma-delimited (declared ``DLM`` COMMA, or commas in every row)."""
    try:
        delimiter = str(las.version.DLM.value).strip().upper()
    except (AttributeError, KeyError):
        delimiter = ""
    return delimiter == "COMMA" or (bool(sample_rows) and all("," in row for row in sample_rows))


def _space_delimited_text(text: str, name: str) -> str:
    """``text`` with its comma-delimited ~A rows written space-delimited.

    Rows with one comma-separated field per ~Curve curve are split on the
    commas (an empty field becomes NaN). Rows with one whitespace-separated
    field per curve use decimal commas, which are read as decimal points.
    Rows that are neither raise ValueError.
    """
    lines = text.splitlines()
    n_curves = 0
    declared = False
    section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("~"):
            title = stripped.upper()
            section = "C" if title.startswith(("~C", "~LOG_DEFINITION")) else title[1:2]
            if _is_data_section(stripped):
                break
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if section == "C" and "." in stripped:
            n_curves += 1
        elif section == "V" and re.match(r"DLM\s*\.\S*\s+COMMA\b", stripped, flags=re.IGNORECASE):
            declared = True
    out: List[str] = []
    in_data = False
    section = ""
    row = 0
    decimal_commas = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("~"):
            in_data = _is_data_section(stripped)
            section = stripped[1:2].upper()
            out.append(line)
            continue
        if not in_data:
            if section == "V" and declared:
                line = re.sub(r"^(\s*DLM\s*\.\S*\s+)COMMA\b", r"\1SPACE", line, flags=re.IGNORECASE)
            out.append(line)
            continue
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        row += 1
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) == n_curves and not any(len(field.split()) > 1 for field in fields):
            out.append(" ".join(field or "NaN" for field in fields))
            continue
        words = stripped.split()
        if len(words) == n_curves and not declared:
            out.append(" ".join(word.replace(",", ".") for word in words))
            decimal_commas = decimal_commas or "," in stripped
            continue
        raise ValueError(
            f"{name}: ~A row {row} has {len(fields)} comma-separated values for the "
            f"{n_curves} ~Curve curves; the comma-delimited data cannot be read."
        )
    if decimal_commas:
        warnings.warn(
            f"{name}: decimal commas in ~A are read as decimal points.",
            UserWarning,
            stacklevel=4,
        )
    return "\n".join(out) + "\n"


def _checked_las(las, path: Path, encoding: Optional[str]):
    """Return ``las`` read again where lasio misread ``path``, and its raw header values.

    A file lasio reads without curves because its section titles are lower
    case (``~curve``, ``~a``) is read with them capitalised. Comma-delimited
    ~A rows (LAS 3.0 ``DLM . COMMA``, or commas in LAS 2.0 rows) are read with
    the comma as delimiter. A file without ~Curve curves or without an ~A
    section raises ValueError.
    """
    text_encoding = encoding or getattr(las, "encoding", None) or "utf-8"
    scan = _scan_las_text(path, text_encoding)
    text = None
    if scan["lowercase_sections"] and not len(las.curves):
        with open(path, "r", encoding=text_encoding, errors="replace") as stream:
            text = re.sub(r"(?m)^(\s*~)([a-z])", lambda m: m.group(1) + m.group(2).upper(), stream.read())
        las = _lasio_read(text, path.name)
    if len(las.curves) and _comma_delimited(las, scan["data"]):
        if text is None:
            with open(path, "r", encoding=text_encoding, errors="replace") as stream:
                text = stream.read()
        las = _lasio_read(_space_delimited_text(text, path.name), path.name)
    if not len(las.curves):
        raise ValueError(
            f"{path.name} has no curves in a ~Curve (~C) section; sections found: "
            f"{', '.join(scan['sections']) or 'none'}."
        )
    if not scan["has_data_section"]:
        raise ValueError(
            f"{path.name} has no ~A (data) section; sections found: {', '.join(scan['sections'])}."
        )
    if not len(las.curves[0].data):
        warnings.warn(f"{path.name}: the ~A section has no data rows.", UserWarning, stacklevel=3)
    return las, _raw_header_values(scan["header"])


def _warn_on_depth_issues(depth: Optional[np.ndarray], source: object = "") -> None:
    """Warn when finite depth samples contain duplicates or change direction."""
    if depth is None:
        return
    arr = np.asarray(depth, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return
    diffs = np.diff(finite)
    issues = []
    if np.any(diffs == 0):
        issues.append("duplicate depth values")
    nonzero_diffs = diffs[diffs != 0]
    if np.any(nonzero_diffs > 0) and np.any(nonzero_diffs < 0):
        issues.append("depth changes direction")
    if issues:
        warnings.warn(
            f"Depth index issue in {source!s}: {', '.join(issues)}. "
            "Verify sample order before interpolation or interval operations.",
            stacklevel=3,
        )


def _normalize_depth_unit(unit: str) -> str:
    normalized = normalize_depth_unit(unit, strict=False, default="ft")
    if normalized not in {"m", "ft"}:
        warnings.warn(
            f"Unrecognized LAS depth unit {normalized!r}; depths are kept as read. "
            "Set well.depth_unit to 'm' or 'ft' if the index is a depth.",
            UserWarning,
            stacklevel=3,
        )
    return normalized


def _read_lasio(path: Path, encoding: Optional[str]):
    """Parse ``path`` with lasio, preferring UTF-8, with readable errors."""
    try:
        if encoding is not None:
            return lasio.read(str(path), encoding=encoding)
        try:
            return lasio.read(str(path), encoding="utf-8", encoding_errors="strict")
        except UnicodeDecodeError:
            return lasio.read(str(path))
    except (KeyError, IndexError, ValueError, lasio.exceptions.LASHeaderError,
            lasio.exceptions.LASDataError) as exc:
        detail = str(exc).strip("'\"") or type(exc).__name__
        raise ValueError(f"{path.name} is not a readable LAS file: {detail}") from exc


class LASReader:
    """
    Reader for LAS (Log ASCII Standard) files.

    Examples
    --------
    >>> reader = LASReader("well_data.las")
    >>> well = reader.to_well()
    """

    def __init__(self, filepath: Union[str, Path], encoding: Optional[str] = None):
        """
        Initialize LAS reader.

        Parameters
        ----------
        filepath : str or Path
            Path to LAS file
        encoding : str, optional
            Text encoding of the file. By default UTF-8 is tried first and
            lasio's encoding detection is used if the file is not UTF-8.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be parsed as LAS, has no ~Curve curves or no
            ~A section. Comma-delimited ~A rows and lower-case section
            titles are read.
        """
        if not HAS_LASIO:
            raise ImportError(
                "lasio is required for LAS file reading. "
                "Install with: pip install lasio"
            )

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"LAS file not found: {filepath}")

        self._las = _read_lasio(self.filepath, encoding)
        self._raw_header: Dict[Tuple[str, str], List[str]] = {}
        if not reproducing_benchmark():
            self._las, self._raw_header = _checked_las(self._las, self.filepath, encoding)

    @property
    def las(self):
        """Return underlying lasio object."""
        return self._las

    @property
    def curves(self) -> List[str]:
        """Return list of curve mnemonics."""
        return [c.mnemonic for c in self._las.curves]

    @property
    def well_name(self) -> str:
        """Return well name from header."""
        return self._las.well.WELL.value if hasattr(self._las.well, 'WELL') else ""

    def get_header_value(self, key: str, default: Any = None) -> Any:
        """Get value from LAS header."""
        try:
            if hasattr(self._las.well, key):
                return getattr(self._las.well, key).value
            if hasattr(self._las.params, key):
                return getattr(self._las.params, key).value
        except (AttributeError, KeyError):
            pass
        return default

    def get_header_unit(self, key: str, default: str = "") -> str:
        """Get unit text from LAS well or parameter header."""
        try:
            if hasattr(self._las.well, key):
                return str(getattr(self._las.well, key).unit or default)
            if hasattr(self._las.params, key):
                return str(getattr(self._las.params, key).unit or default)
        except (AttributeError, KeyError):
            pass
        return default

    def get_curve_data(self, mnemonic: str) -> Optional[np.ndarray]:
        """Get data for a specific curve."""
        if mnemonic in self.curves:
            return np.asarray(self._las[mnemonic], dtype=float)
        return None

    def get_curve_unit(self, mnemonic: str) -> str:
        """Get unit for a specific curve."""
        for curve in self._las.curves:
            if curve.mnemonic == mnemonic:
                return curve.unit or ""
        return ""

    def to_well(self) -> "Well":
        """
        Convert LAS data to Well object.

        The first ~Curve curve is the depth index (DEPT, TDEP, INDEX, ...),
        as LAS 2.0 defines; a later TVD or DEPTH curve is an ordinary curve.
        Data rows whose index is null (the header NULL, -999.25 or NaN) are
        left out with a warning, and an unreadable index value raises
        ValueError naming its data row. In the curves, unreadable tokens
        (``-1.#IND``, ``****``) are read as NaN with a warning, Fortran ``D``
        exponents are read, and values of :data:`COMMON_LAS_NULLS` that differ
        from the header NULL are read as NaN with a warning.

        Header values with units are stored as:

        - ``header.td``: total depth in the well's depth unit (``well.depth_unit``),
          converted from the TD unit; ``header.td_source_unit`` is the TD unit
          as written.
        - ``header.bit_size``: bit size in inches, converted from mm, cm or m;
          ``header.bit_size_source_unit`` is the BS unit as written. Without
          a BS item, numbered bit runs (BS1, BS2, ... with the depths BSD1,
          BSD2, ... they drilled) give ``header.bit_size_intervals``.

        A TD or BS whose unit is not recognised is left unset (None) with a
        warning; one without a unit is taken to be in the depth unit or in
        inches. Other ~Well items are kept in ``header.las_well_items`` and
        the ~Other text in the ``LAS_OTHER`` parameter, which
        :func:`petrophys.io.export_las` writes back. Numeric-looking header
        values written with leading zeros (``LIC . 00123``) are kept as text.

        Returns
        -------
        Well
            Well object with loaded data
        """
        from petrophys.model.well import Well
        from petrophys.model.curve import Curve, CurveMetadata
        from petrophys.model.data_types import DataType

        historical = reproducing_benchmark()
        well = Well(name=self.well_name)

        # Set header information
        well.header.well_name = self.well_name
        well.header.uwi = self.get_header_value("UWI", "")
        well.header.api = self.get_header_value("API", "")
        well.header.field = self.get_header_value("FLD", "")
        well.header.company = self.get_header_value("COMP", "")
        well.header.location = self.get_header_value("LOC", "")
        well.header.county = self.get_header_value("CNTY", "")
        well.header.state = self.get_header_value("STAT", "")
        well.header.country = self.get_header_value("CTRY", "")

        # Get numeric header values
        well.header.kb_elevation = self._parse_float(self.get_header_value("EKBF"))
        well.header.td = self._parse_float(self.get_header_value("TD"))
        well.header.bht = self._parse_float(self.get_header_value("BHT"))
        well.header.bht_unit = self.get_header_unit("BHT", well.header.bht_unit) or well.header.bht_unit
        well.header.rm = self._parse_float(self.get_header_value("RM"))
        well.header.rmf = self._parse_float(self.get_header_value("RMF"))
        well.header.rmc = self._parse_float(self.get_header_value("RMC"))
        well.header.bit_size = self._parse_float(self.get_header_value("BS"))

        # Persist ~Parameter metadata on the Well so workflows/tutorials can
        # access zoned petrophysical inputs such as RW_Z1, RMF_Z1, BHT_Z1, etc.
        self._load_parameters(well)

        # Set depth
        depth_curve = None
        if historical:
            for name in ["DEPT", "DEPTH", "MD", "TVD"]:
                if name in self.curves:
                    depth_curve = name
                    break
        elif self.curves:
            # LAS 2.0: the first ~C curve is the index (e.g. DEPT, TDEP, INDEX).
            depth_curve = self.curves[0]

        keep = None
        if depth_curve:
            if historical:
                well._depth = self._las[depth_curve]
            else:
                well._depth, keep = self._index_values(self._las.curves[0])
            # Get depth unit
            for curve in self._las.curves:
                if curve.mnemonic == depth_curve:
                    index_unit = curve.unit
                    if not (index_unit or "").strip() and not reproducing_benchmark():
                        # A blank index unit: STRT/STOP/STEP carry the index unit.
                        index_unit = self._well_section_depth_unit()
                        if not index_unit:
                            warnings.warn(
                                f"{self.filepath.name}: the index curve {depth_curve!r} "
                                "and STRT/STOP/STEP have no depth unit; feet are assumed. "
                                "Set well.depth_unit if the depths are in metres.",
                                UserWarning,
                                stacklevel=2,
                            )
                    well._depth_unit = _normalize_depth_unit(index_unit)
                    section_unit = self._well_section_depth_unit()
                    if section_unit and (curve.unit or "").strip() and not reproducing_benchmark():
                        section = normalize_depth_unit(section_unit, strict=False)
                        if {section, well._depth_unit} == {"m", "ft"}:
                            warnings.warn(
                                f"{self.filepath.name}: the index curve {depth_curve!r} is in "
                                f"{well._depth_unit} but STRT/STOP/STEP are in {section}; "
                                f"{well._depth_unit} is used. Set well.depth_unit if that is wrong.",
                                UserWarning,
                                stacklevel=2,
                            )
                    # The label as written, which the "geolink" profile writes back.
                    well._source_depth_unit_label = (curve.unit or "").strip()
                    break
            _warn_on_depth_issues(well._depth, self.filepath)

        if not historical:
            self._convert_unit_header_values(well)
            self._load_other_header(well)

        # Add curves
        null = self._null_value()
        unreadable: Dict[str, List[Tuple[int, str]]] = {}
        undeclared_nulls: Dict[str, int] = {}
        for las_curve in self._las.curves:
            if las_curve.mnemonic == depth_curve:
                continue

            metadata = CurveMetadata(
                description=las_curve.descr or "",
                source="LAS",
                data_type=DataType.CURVE
            )

            data = las_curve.data
            if not historical:
                data, bad = _numeric_values(data, null)
                if data is None:
                    warnings.warn(
                        f"Skipping non-numeric LAS curve {las_curve.mnemonic!r}: "
                        "data could not be converted to float."
                    )
                    continue
                if bad:
                    unreadable[las_curve.mnemonic] = bad
                if keep is not None:
                    data = data[keep]
                sentinel = np.zeros(data.shape, dtype=bool)
                for value in COMMON_LAS_NULLS:
                    if value != null:
                        sentinel |= data == value
                if sentinel.any():
                    data = np.where(sentinel, np.nan, data)
                    undeclared_nulls[las_curve.mnemonic] = int(sentinel.sum())

            try:
                curve = Curve(
                    mnemonic=las_curve.mnemonic,
                    data=data,
                    unit=las_curve.unit or "",
                    depth=well._depth,
                    depth_unit=well._depth_unit,
                    metadata=metadata
                )
            except ValueError:
                warnings.warn(
                    f"Skipping non-numeric LAS curve {las_curve.mnemonic!r}: "
                    "data could not be converted to float."
                )
                continue

            well._curves[las_curve.mnemonic.upper()] = curve

        if unreadable:
            details = "; ".join(
                f"{mnemonic}: {len(bad)} ({', '.join(sorted({token for _, token in bad})[:3])}; "
                f"first at data row {bad[0][0] + 1})"
                for mnemonic, bad in unreadable.items()
            )
            warnings.warn(
                f"{self.filepath.name}: unreadable ~A values are read as missing (NaN) - {details}",
                UserWarning,
                stacklevel=2,
            )
        if undeclared_nulls:
            header_null = "no NULL" if null is None else f"NULL {null:g}"
            details = ", ".join(f"{mnemonic}: {count}" for mnemonic, count in undeclared_nulls.items())
            warnings.warn(
                f"{self.filepath.name}: values {', '.join(f'{v:g}' for v in COMMON_LAS_NULLS)} "
                f"are read as missing (NaN) although the header has {header_null} - {details}",
                UserWarning,
                stacklevel=2,
            )

        return well

    def _null_value(self) -> Optional[float]:
        """The header NULL value, or None when the header has none."""
        try:
            return float(self._las.well.NULL.value)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _index_values(self, item) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """The index curve as float, and the mask of the data rows kept (None for all).

        Rows whose index is null (the header NULL, -999.25 or NaN) are left
        out with a warning; an unreadable index value raises ValueError.
        """
        values, bad = _numeric_values(item.data, None)
        if bad:
            row, token = bad[0]
            raise ValueError(
                f"{self.filepath.name}: the index curve {item.mnemonic!r} has the unreadable "
                f"value {token!r} in data row {row + 1}; correct or remove that row."
            )
        null = self._null_value()
        if values is None:
            raise ValueError(f"{self.filepath.name}: the index curve {item.mnemonic!r} has no numeric values")
        missing = ~np.isfinite(values)
        for value in {null, -999.25} - {None}:
            missing |= values == value
        if not missing.any():
            return values, None
        warnings.warn(
            f"{self.filepath.name}: {int(missing.sum())} data row(s) with a null index "
            f"{item.mnemonic!r} value are left out (first at data row {int(np.argmax(missing)) + 1}).",
            UserWarning,
            stacklevel=3,
        )
        return values[~missing], ~missing

    def _convert_unit_header_values(self, well: "Well") -> None:
        """Store header TD in the well's depth unit and bit size in inches."""
        unit = self.get_header_unit("TD").strip()
        well.header.td_source_unit = unit
        if well.header.td is not None and unit:
            source = normalize_depth_unit(unit, strict=False)
            if source not in {"m", "ft"}:
                warnings.warn(
                    f"{self.filepath.name}: header TD unit {unit!r} is not recognised; "
                    "header.td is left unset.",
                    UserWarning,
                    stacklevel=3,
                )
                well.header.td = None
            elif well.depth_unit in {"m", "ft"} and source != well.depth_unit:
                well.header.td = well.header.td * 0.3048 if source == "ft" else well.header.td / 0.3048

        unit = self.get_header_unit("BS").strip()
        well.header.bit_size_source_unit = unit
        if well.header.bit_size is not None and unit:
            factor = _INCHES_PER_UNIT.get(unit.lower().replace(".", "").replace(" ", ""))
            if factor is None:
                warnings.warn(
                    f"{self.filepath.name}: header BS (bit size) unit {unit!r} is not "
                    "recognised; header.bit_size is left unset.",
                    UserWarning,
                    stacklevel=3,
                )
                well.header.bit_size = None
            else:
                well.header.bit_size = round(well.header.bit_size * factor, 6)
        if well.header.bit_size is None:
            self._set_numbered_bit_sizes(well)

    def _set_numbered_bit_sizes(self, well: "Well") -> None:
        """Bit sizes of numbered bit runs (BS1, BS2, ...) over the depths each drilled.

        Some service companies write each bit run's size as BS1, BS2, ... and
        the depth it drilled to as BSD1, BSD2, ... Bit k covers the hole from
        the previous bit's depth down to BSDk, in ``header.bit_size_intervals``
        as (top, bottom, inches) in the well's depth unit. A single bit covers
        the whole well and is also ``header.bit_size``. Several bits without a
        depth for each cannot be placed and are left unset with a warning.
        """
        bits = []
        for section in (self._las.well, self._las.params):
            for item in section:
                match = re.fullmatch(r"BS(\d+)", str(item.mnemonic).strip().upper())
                if match is None:
                    continue
                size = self._parse_float(item.value)
                if size is None:
                    continue
                unit = str(item.unit or "").strip()
                factor = (
                    _INCHES_PER_UNIT.get(unit.lower().replace(".", "").replace(" ", ""))
                    if unit else 1.0
                )
                if factor is None:
                    continue
                depth = self._bit_depth(well, f"BSD{match.group(1)}")
                bits.append((size * factor, depth, item.mnemonic))
        if not bits:
            return
        if len(bits) == 1:
            size, depth, _ = bits[0]
            well.header.bit_size = round(size, 6)
            bottom = np.inf if depth is None else depth
            well.header.bit_size_intervals = [(-np.inf, bottom, round(size, 6))]
            return
        if any(depth is None for _, depth, _ in bits):
            warnings.warn(
                f"{self.filepath.name}: bit sizes {', '.join(name for _, _, name in bits)} lack the "
                "depths each bit drilled (BSD1, BSD2, ...), so the bit size of each depth is "
                "unknown and none is used.",
                UserWarning,
                stacklevel=4,
            )
            return
        top = -np.inf
        intervals = []
        located_bits = [(size, depth, name) for size, depth, name in bits if depth is not None]
        for size, depth, _ in sorted(located_bits, key=lambda bit: bit[1]):
            intervals.append((top, depth, round(size, 6)))
            top = depth
        well.header.bit_size_intervals = intervals

    def _bit_depth(self, well: "Well", key: str) -> Optional[float]:
        """Header depth ``key`` in the well's depth unit (a blank unit is that unit), or None."""
        depth = self._parse_float(self.get_header_value(key))
        if depth is None:
            return None
        unit = self.get_header_unit(key).strip()
        source = normalize_depth_unit(unit, strict=False) if unit else well.depth_unit
        if source not in {"m", "ft"} or well.depth_unit not in {"m", "ft"}:
            return None if unit else depth
        if source != well.depth_unit:
            depth = depth * 0.3048 if source == "ft" else depth / 0.3048
        return depth

    def _header_value(self, section: str, item: Any, occurrence: int) -> Any:
        """``item.value``, or its text as written when lasio read a number written with leading zeros."""
        value = getattr(item, "value", None)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            mnemonic = str(getattr(item, "original_mnemonic", "") or getattr(item, "mnemonic", "")).upper()
            written = self._raw_header.get((section, mnemonic), [])
            if occurrence < len(written) and _LEADING_ZERO_NUMBER.match(written[occurrence]):
                return written[occurrence]
        return value

    def _load_other_header(self, well: "Well") -> None:
        """Keep the ~Well items not mapped to header fields and the ~Other text."""
        seen: Dict[str, int] = {}
        for item in self._las.well:
            original = str(getattr(item, "original_mnemonic", "") or item.mnemonic).upper()
            occurrence = seen.get(original, 0)
            seen[original] = occurrence + 1
            mnemonic = str(item.mnemonic or "").strip()
            if not mnemonic or mnemonic.upper() in _MAPPED_WELL_ITEMS:
                continue
            value = self._header_value("W", item, occurrence)
            if value is None or not str(value).strip():
                continue
            if isinstance(value, np.generic):
                value = value.item()
            well.header.las_well_items[mnemonic] = {
                "unit": str(item.unit or ""),
                "value": value,
                "descr": str(item.descr or ""),
            }
        other = str(self._las.other or "").strip()
        if other and well.get_parameter("LAS_OTHER") is None:
            well.set_parameter("LAS_OTHER", other)

    def _load_parameters(self, well: "Well") -> None:
        """Copy LAS ~Parameter entries into ``well`` parameters."""
        try:
            params = list(self._las.params)
        except Exception:
            params = []

        historical = reproducing_benchmark()
        seen: Dict[str, int] = {}
        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic:
                continue

            raw_value = getattr(item, "value", None)
            if not historical:
                original = str(getattr(item, "original_mnemonic", "") or mnemonic).upper()
                occurrence = seen.get(original, 0)
                seen[original] = occurrence + 1
                raw_value = self._header_value("P", item, occurrence)
            if not historical and isinstance(raw_value, str) and _LEADING_ZERO_NUMBER.match(raw_value):
                parsed = None  # an identifier such as a licence number 00123
            else:
                parsed = self._parse_float(raw_value)
            value = parsed if parsed is not None else raw_value
            well.set_parameter(mnemonic, value)

            unit = str(getattr(item, "unit", "") or "").strip()
            descr = str(getattr(item, "descr", "") or "").strip()
            if unit:
                well.set_parameter(f"{mnemonic}__UNIT", unit)
            if descr:
                well.set_parameter(f"{mnemonic}__DESCR", descr)

        # Promote single-valued zoned metadata into the header when the base
        # header is blank, so existing code can benefit without knowing about
        # the zoned LAS convention.
        if well.header.bht is None:
            bht = self._get_constant_zoned_float("BHT")
            bht_unit = self._get_constant_zoned_unit("BHT").lower()
            if bht is not None and "c" in bht_unit and "f" not in bht_unit:
                bht = (bht * 9.0 / 5.0) + 32.0
            elif bht is not None and not reproducing_benchmark():
                unit_key = bht_unit.replace("deg", "").replace("\u00b0", "").replace(" ", "").strip(".")
                if unit_key in {"k", "kelvin"}:
                    bht = (bht - 273.15) * 9.0 / 5.0 + 32.0
                elif not unit_key:
                    warnings.warn(
                        f"{self.filepath.name}: zoned BHT has no unit; degF is assumed.",
                        UserWarning,
                        stacklevel=3,
                    )
                elif "f" not in unit_key:
                    warnings.warn(
                        f"{self.filepath.name}: zoned BHT unit {bht_unit!r} is not "
                        "recognised; the header BHT is left unset.",
                        UserWarning,
                        stacklevel=3,
                    )
                    bht = None
            if bht is not None:
                well.header.bht = bht
                well.header.bht_unit = "F"

        if well.header.rmf is None:
            rmf = self._get_constant_zoned_float("RMF")
            if rmf is not None:
                well.header.rmf = rmf

        if not well.header.field:
            field_name = self._get_constant_zoned_text("FLD")
            if field_name:
                well.header.field = field_name

    def _well_section_depth_unit(self) -> str:
        """Return the single non-blank unit of ~W STRT/STOP/STEP, or ''."""
        units = {self.get_header_unit(key).strip() for key in ("STRT", "STOP", "STEP")}
        units.discard("")
        return next(iter(units)) if len(units) == 1 else ""

    def _get_constant_zoned_float(self, prefix: str) -> Optional[float]:
        """Return a single unique float value from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            parsed = self._parse_float(getattr(item, "value", None))
            if parsed is not None:
                values.add(float(parsed))

        if len(values) == 1:
            return next(iter(values))
        return None

    def _get_constant_zoned_text(self, prefix: str) -> str:
        """Return a single unique text value from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            value = str(getattr(item, "value", "") or "").strip()
            if value:
                values.add(value)

        if len(values) == 1:
            return next(iter(values))
        return ""

    def _get_constant_zoned_unit(self, prefix: str) -> str:
        """Return a single unique unit string from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            value = str(getattr(item, "unit", "") or "").strip()
            if value:
                values.add(value)

        if len(values) == 1:
            return next(iter(values))
        return ""

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        """Parse value to float, return None if not possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def read_las(
    filepath: Union[str, Path],
    *,
    encoding: Optional[str] = None,
    **kwargs
) -> "Well":
    """
    Read LAS file and return Well object.

    Parameters
    ----------
    filepath : str or Path
        Path to LAS file
    encoding : str, optional
        Text encoding of the file (e.g. ``"latin-1"``). By default UTF-8 is
        tried first, then lasio's encoding detection.
    **kwargs
        Reserved for future use; unsupported arguments raise TypeError.

    Returns
    -------
    Well
        Well object with loaded data

    Examples
    --------
    >>> from petrophys.io import read_las
    >>> well = read_las("well_data.las")
    >>> print(well.curves)
    ['GR', 'RHOB', 'NPHI', 'RT']
    """
    if kwargs:
        raise TypeError(f"Unsupported LAS reader options: {', '.join(sorted(kwargs))}")
    reader = LASReader(filepath, encoding=encoding)
    return reader.to_well()
