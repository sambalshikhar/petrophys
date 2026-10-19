"""
Export functionality for well log data.

This module provides functions to export Well data to various
file formats including LAS, CSV, and Excel.
"""

from typing import TYPE_CHECKING, Optional, Union, List, Dict, Any
from pathlib import Path
import warnings
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


ML_IMPUTATION_NOTES_PARAMETER = "ML_IMPUTATION_NOTES"
VALUE_RANGE_QC_PARAMETER = "VALUE_RANGE_QC_REPORT"


def _export_well_other_comments(las, well: "Well") -> None:
    """Write workflow comments into the LAS ~Other section."""
    lines = []
    existing_other = ""
    if hasattr(well, "get_parameter"):
        existing_other = well.get_parameter("LAS_OTHER", "") or ""
    if existing_other:
        lines.extend(str(existing_other).splitlines())

    notes: List[Any] = []
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
        if name in {ML_IMPUTATION_NOTES_PARAMETER, VALUE_RANGE_QC_PARAMETER, "LAS_OTHER"}:
            continue

        unit = parameters.get(f"{name}__UNIT", "")
        descr = parameters.get(f"{name}__DESCR", "")

        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = ""
        if isinstance(value, (dict, list, tuple, set, np.ndarray)) and not reproducing_benchmark():
            # Structured metadata (e.g. dlis_merge) has no LAS ~P form; its
            # text would re-read truncated at the first colon.
            continue

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
    depth_unit_label: Optional[str] = None,
    pad_header_by_bytes: bool = False,
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
    depth_unit_label : str, optional
        Depth unit label to write for DEPT, STRT, STOP and STEP instead of
        ``well.depth_unit``, e.g. ``"FT"`` as a source file wrote it. It must
        name the same unit.
    pad_header_by_bytes : bool, default False
        Pad the header columns by each text's UTF-8 byte length rather than
        its character count, as the GeoLink eval package's interpreted LAS
        were padded (a field name such as SLEIPNER OST with an O-slash then
        widens its column by one). The bytes written are the same UTF-8.
    **kwargs
        Additional arguments passed to lasio's LASFile.write
        (e.g. ``fmt="%.8f"`` to control numeric precision). Without
        ``fmt`` the values are written with six significant figures
        (``%.6g``), or lasio's fixed five decimals when reproducing a
        benchmark context package.

    Notes
    -----
    Irregularly sampled depths are written with ``STEP 0`` as LAS 2.0
    requires, and samples without a finite depth are left out (a LAS index
    cannot be null). Non-scalar well parameters (dicts, lists) are not written.

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
    historical = reproducing_benchmark()
    # Six significant figures keep small values such as permeability in mD;
    # the benchmark context profiles keep the packages' five fixed decimals.
    kwargs.setdefault("fmt", "%.5f" if historical else "%.6g")
    depth_label = well.depth_unit
    if depth_unit_label:
        from petrophys.model.units import normalize_depth_unit

        if normalize_depth_unit(depth_unit_label, strict=False) != well.depth_unit:
            raise ValueError(
                f"depth_unit_label {depth_unit_label!r} does not name the well's depth unit "
                f"{well.depth_unit!r}"
            )
        depth_label = depth_unit_label

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
    if not historical:
        for mnemonic, value, descr in (
            ("CNTY", header.county, "COUNTY"),
            ("STAT", header.state, "STATE"),
            ("CTRY", header.country, "COUNTRY"),
        ):
            if value:
                setattr(las.well, mnemonic, lasio.HeaderItem(mnemonic=mnemonic, value=value, descr=descr))

    # A LAS index is never null: samples without a finite depth are left out.
    keep = None
    irregular = False
    step_value = well.step
    if well.depth is not None and not historical:
        depth_values = np.asarray(well.depth, dtype=float)
        finite_depth = np.isfinite(depth_values)
        if not finite_depth.all():
            keep = finite_depth
            warnings.warn(
                f"export_las: {int((~finite_depth).sum())} sample(s) without a finite "
                "depth are not written.",
                UserWarning,
                stacklevel=2,
            )
        diffs = np.diff(depth_values[finite_depth])
        if diffs.size > 1 and not np.allclose(
            diffs, diffs[0], rtol=0.0, atol=1e-3 * abs(diffs[0])
        ):
            irregular = True  # LAS 2.0: STEP 0 for variable sampling

    # Set depth reference
    if well.top is not None:
        las.well.STRT = lasio.HeaderItem(
            mnemonic="STRT",
            unit=depth_label,
            value=well.top,
            descr="START DEPTH"
        )
    if well.bottom is not None:
        las.well.STOP = lasio.HeaderItem(
            mnemonic="STOP",
            unit=depth_label,
            value=well.bottom,
            descr="STOP DEPTH"
        )
    if step_value is not None:
        las.well.STEP = lasio.HeaderItem(
            mnemonic="STEP",
            unit=depth_label,
            value=step_value,
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
            well.depth if keep is None else np.asarray(well.depth, dtype=float)[keep],
            unit=depth_label,
            descr="DEPTH"
        )

    # Get curves to export
    if curves is None:
        curves = well.curves
    else:
        curves = [c.upper() for c in curves]

    missing = [mnemonic for mnemonic in curves if mnemonic not in well._curves]
    if missing and not historical:
        warnings.warn(
            f"export_las: requested curves not in the well are not written: {', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )

    # Add curves
    for mnemonic in curves:
        if mnemonic not in well._curves:
            continue
        curve = well._curves[mnemonic]

        # Replace non-finite values (NaN, +/-Inf) with the LAS null value, so
        # the file reads back with the same missing samples.
        data = curve.data.copy().astype(float)
        data = np.where(np.isfinite(data), data, -999.25)
        if keep is not None:
            data = data[keep]

        las.append_curve(
            mnemonic,
            data,
            unit=curve.unit,
            descr=curve.metadata.description
        )

    if irregular and kwargs.get("STEP") is None:
        # lasio otherwise writes the first depth difference as STEP.
        kwargs["STEP"] = kwargs["fmt"] % 0.0

    # Write file
    # lasio would open the file with the platform's locale encoding, which
    # writes non-ASCII header text (e.g. a field name like OSEBERG OST) in a
    # single-byte code page. Write UTF-8, as the LAS files we read use.
    if pad_header_by_bytes:
        # Each UTF-8 byte as one latin-1 character: lasio then pads by bytes,
        # and writing latin-1 reproduces the UTF-8 bytes.
        def as_bytes(text: Any) -> Any:
            return text.encode("utf-8").decode("latin-1") if isinstance(text, str) else text

        for section in (las.version, las.well, las.curves, las.params):
            for item in section:
                item.value = as_bytes(item.value)
                item.descr = as_bytes(item.descr)
                item.unit = as_bytes(item.unit)
        las.other = as_bytes(las.other)
        with open(filepath, "w", encoding="latin-1") as handle:
            las.write(handle, version=version, **kwargs)
        return
    with open(filepath, "w", encoding="utf-8") as handle:
        las.write(handle, version=version, **kwargs)


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
        Output file path. The file is written as .xlsx; another suffix
        (including .xls) is replaced by .xlsx.
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
    if filepath.suffix.lower() != '.xlsx':
        # openpyxl writes xlsx content only; a .xls name would mislabel it.
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
            header_data["depth_unit"] = well.depth_unit
            header_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in header_data.items()
                if v is not None and v != ""
            ])
            header_df.to_excel(writer, sheet_name="Header", index=False)

        # Write log data
        df = well.to_dataframe(curves=curves, include_depth=include_depth)
        # Non-finite values as empty cells, as export_csv/export_las null them.
        df = df.replace([np.inf, -np.inf], np.nan)
        df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)

        # Write curve info
        curve_info = []
        for mnemonic in curves:
            if mnemonic in well._curves:
                curve = well._curves[mnemonic]
                values = np.asarray(curve.data, dtype=float)
                finite = values[np.isfinite(values)]
                curve_info.append({
                    "Mnemonic": mnemonic,
                    "Unit": curve.unit,
                    "Description": curve.metadata.description,
                    "Min": float(finite.min()) if finite.size else float("nan"),
                    "Max": float(finite.max()) if finite.size else float("nan"),
                    "Valid Count": int(finite.size),
                    "Null Count": int(values.size - finite.size)
                })
        if curve_info:
            curve_df = pd.DataFrame(curve_info)
            curve_df.to_excel(writer, sheet_name="Curve Info", index=False)
