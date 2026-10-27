"""Shared, explicit table import for formation and lithology context."""

from collections.abc import Mapping
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from petrophys.exceptions import DataQualityWarning
from petrophys.model.units import normalize_depth_unit


CONTEXT_COLUMNS = {
    "well": ("well", "wellname", "wellidentifier", "wellid", "uwi", "api"),
    "depth_unit": ("depthunit", "depthunits", "unit", "units"),
    "depth_reference": ("depthreference", "depthtype", "reference"),
    "datum": ("datum", "depthdatum"),
}


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _well_key(value):
    return re.sub(r"\s+", "", str(value).strip().upper()).replace("_", "/")


def _unit(value):
    return normalize_depth_unit(value)


def _read_table(source, *, sheet_name=None, sep=None, encoding="utf-8-sig"):
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    if not isinstance(source, (str, Path)):
        return pd.DataFrame(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Geology input not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        with pd.ExcelFile(path) as workbook:
            if sheet_name is None and len(workbook.sheet_names) != 1:
                raise ValueError(
                    f"Select sheet_name explicitly; workbook sheets: {workbook.sheet_names}"
                )
            return pd.read_excel(
                workbook, sheet_name=sheet_name if sheet_name is not None else 0,
                dtype=str, keep_default_na=False,
            )
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError("Geology inputs require CSV, TSV, text, Excel, or a DataFrame.")
    delimiter = sep if sep is not None else ("\t" if path.suffix.lower() == ".tsv" else ",")
    try:
        return pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Geology input is empty; provide a header and data rows.") from exc


def _prepare_table(source, aliases, required, *, columns=None, well_name=None,
                   well=None, depth_unit=None, depth_reference=None, datum=None,
                   sheet_name=None, sep=None, encoding="utf-8-sig", strict_columns=False):
    table = _read_table(source, sheet_name=sheet_name, sep=sep, encoding=encoding)
    inherited = dict(table.attrs)
    if table.empty:
        raise ValueError("Geology input contains no data rows; fill the template first.")
    normalized = [_key(column) for column in table.columns]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate geology column names after normalizing whitespace/case.")
    lookup = dict(zip(normalized, table.columns))
    # "MD (ft)", "Top [m]": the name without its bracketed unit also matches,
    # and the unit becomes the depth unit when none is given.
    bracket_units = {}
    for column in table.columns:
        match = re.match(r"^\s*(.*?)\s*[\(\[]\s*([^\)\]]*)\s*[\)\]]\s*$", str(column))
        if match and _key(match.group(1)) and _key(match.group(1)) not in lookup:
            lookup[_key(match.group(1))] = column
            bracket_units[column] = match.group(2).strip()
    mapping = {}
    for canonical, candidates in {**aliases, **CONTEXT_COLUMNS}.items():
        explicit = (columns or {}).get(canonical)
        matches = ([explicit] if explicit in table.columns else []) if explicit is not None else [
            lookup[name] for name in dict.fromkeys((_key(canonical), *candidates)) if name in lookup
        ]
        if explicit is not None and not matches:
            raise ValueError(f"Mapped column {explicit!r} for {canonical!r} was not found.")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous columns for {canonical}: {matches}; pass columns explicitly.")
        if matches:
            mapping[matches[0]] = canonical
    if depth_unit is None:
        units = {bracket_units[c] for c, canonical in mapping.items()
                 if canonical in ("top_depth", "bottom_depth") and c in bracket_units}
        if len(units) == 1:
            try:
                depth_unit = normalize_depth_unit(units.pop(), strict=True)
            except ValueError:
                depth_unit = None
    unknown_mappings = set(columns or {}) - set(aliases) - set(CONTEXT_COLUMNS)
    if unknown_mappings:
        raise ValueError(f"Unknown canonical column mappings: {sorted(unknown_mappings)}")
    extras = [str(column) for column in table.columns if column not in mapping]
    if strict_columns and extras:
        raise ValueError(f"Unsupported lithology columns {extras}; use the input template or explicit mappings.")
    table = table.rename(columns=mapping)[list(mapping.values())].reset_index(drop=True)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(
            f"Missing geology columns {sorted(missing)}. Expected {list(required)}; "
            "use the input templates or pass columns and sep explicitly."
        )
    input_rows = len(table)
    notes = []
    selected_well = None
    if "well" in table:
        identifiers = table["well"].fillna("").astype(str).str.strip()
        if (identifiers == "").any():
            raise ValueError("Missing well identifiers in geology table; select and identify every row.")
        targets = {_well_key(well_name)} if well_name is not None else set()
        if well is not None and not targets:
            header = well.header
            targets = {_well_key(value) for value in (well.name, header.well_name, header.uwi, header.api) if value}
        unique = identifiers.unique().tolist()
        if not targets and len(unique) > 1:
            raise ValueError(f"Multiple wells in geology input: {unique}; supply well_name or well.")
        if targets:
            table = table.loc[identifiers.map(_well_key).isin(targets)]
            if table.empty:
                raise ValueError(f"No matching geology rows for well {well_name or well.name!r}; available: {unique}")
            if table["well"].map(_well_key).nunique() > 1:
                raise ValueError("Several well identifiers match; select well_name explicitly.")
        selected_well = str(table["well"].iloc[0])

    metadata = {}
    for field, declared in (("depth_unit", depth_unit), ("depth_reference", depth_reference), ("datum", datum)):
        values = []
        if field in table:
            if table[field].isna().any() or table[field].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing {field} metadata in some geology rows.")
            values = table[field].astype(str).str.strip().unique().tolist()
        if inherited.get(field) is not None:
            values.append(inherited[field])
        if declared is not None:
            values.append(declared)
        values = [_unit(value) if field == "depth_unit" else str(value).strip().upper() for value in values]
        if len(set(values)) > 1:
            raise ValueError(f"Conflicting {field} values: {values}; align inputs before import.")
        metadata[field] = values[0] if values else None

    if well is not None:
        for field in ("depth_reference", "datum"):
            target = well.get_parameter(field)
            source_value = metadata[field]
            if source_value and not target:
                raise ValueError(f"Geology declares {field}={source_value}; set well.set_parameter('{field}', ...) before joining.")
            if source_value and str(target).strip().upper() != source_value:
                raise ValueError(f"Incompatible {field}: geology={source_value}, well={target}; a survey/datum transformation is required.")
            if not source_value:
                notes.append(f"No geology {field} supplied; assumed to match the well (not independently verified).")
                metadata[field] = target

    report = {"input_rows": input_rows, "accepted_rows": len(table),
              "filtered_rows": input_rows - len(table), "well": selected_well,
              "column_mapping": {str(key): value for key, value in mapping.items()},
              "ignored_columns": extras, "diagnostics": notes}
    metadata["import_report"] = report
    table.attrs = metadata
    return table


def _validate_depths(table, fields, *, well=None):
    for field in fields:
        if field not in table:
            continue
        numeric = pd.to_numeric(table[field], errors="coerce").astype(float)
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            row = int(table.index[invalid][0]) + 2
            raise ValueError(f"Geology row {row}: {field} must be a finite number.")
        table[field] = numeric
    unit = table.attrs.get("depth_unit")
    if well is not None:
        target_unit = _unit(well.depth_unit)
        factor = 1.0
        if unit and unit != target_unit:
            factor = 0.3048 if unit == "ft" else 1.0 / 0.3048
            for field in fields:
                if field in table:
                    table[field] *= factor
            table.attrs["import_report"]["depth_conversion"] = {"from": unit, "to": target_unit, "factor": factor}
        elif not unit:
            table.attrs["import_report"]["diagnostics"].append(
                f"No geology depth_unit supplied; assumed {target_unit} to match the well."
            )
        table.attrs["depth_unit"] = target_unit
        if "depth_unit" in table:
            table["depth_unit"] = target_unit
        _check_depth_coverage(table, fields, well, target_unit, unit_assumed=not unit)
    return table


def _check_depth_coverage(table, fields, well, unit, *, unit_assumed):
    """Report geology outside the logged interval; warn when none overlaps it."""
    depth = np.asarray(well.depth if well.depth is not None else [], dtype=float)
    depth = depth[np.isfinite(depth)]
    if depth.size == 0 or table.empty:
        return
    log_top, log_base = float(depth.min()), float(depth.max())
    tops = table[fields[0]].to_numpy(dtype=float)
    bases = table[fields[1]].to_numpy(dtype=float) if len(fields) > 1 and fields[1] in table else tops
    bases = np.where(np.isfinite(bases), bases, tops)
    outside = (bases < log_top) | (tops > log_base)
    if not outside.any():
        return
    report = table.attrs["import_report"]
    report["outside_logged_interval"] = int(outside.sum())
    report["diagnostics"].append(
        f"{int(outside.sum())} of {len(table)} geology rows lie outside the logged interval "
        f"{log_top:g}-{log_base:g} {unit}."
    )
    if not outside.all():
        if unit_assumed:
            # Partly outside with an assumed unit: say so when the other unit
            # would place more of the rows inside the logged interval.
            other = "m" if unit == "ft" else "ft"
            factor = 0.3048 if other == "m" else 1.0 / 0.3048
            outside_other = (bases / factor < log_top) | (tops / factor > log_base)
            if outside_other.sum() < outside.sum():
                warnings.warn(
                    f"{int(outside.sum())} of {len(table)} geology rows lie outside the logged "
                    f"interval {log_top:g}-{log_base:g} {unit}, with no geology depth_unit "
                    f"supplied ({unit} assumed); {int(outside_other.sum())} would if the depths "
                    f"are in {other}. Pass depth_unit for the geology file.",
                    DataQualityWarning,
                    stacklevel=4,
                )
        return
    message = (
        f"None of the {len(table)} geology rows overlaps the logged interval "
        f"{log_top:g}-{log_base:g} {unit}"
    )
    if unit_assumed:
        other = "m" if unit == "ft" else "ft"
        factor = 0.3048 if other == "m" else 1.0 / 0.3048
        converted_tops, converted_bases = tops / factor, bases / factor
        if (~((converted_bases < log_top) | (converted_tops > log_base))).any():
            message += (
                f"; the depths fit if they are in {other}. "
                f"Pass depth_unit={other!r} for the geology file"
            )
        else:
            message += "; check the geology depth unit, reference and datum"
    else:
        message += "; check the geology depth reference and datum"
    warnings.warn(message + ".", DataQualityWarning, stacklevel=4)


def read_formations(source, *, columns=None, well_name=None, well=None, depth_unit=None,
                    depth_reference=None, datum=None, sheet_name=None, sep=None,
                    encoding="utf-8-sig", rank=None) -> pd.DataFrame:
    """Read formation picks or bounded intervals into one canonical table.

    Required columns are ``formation_name,top_depth``; ``bottom_depth`` is
    optional and is never invented. CSV/TSV/Excel, DataFrames, FormationTops,
    interval lists, name/depth mappings and pairs are supported. ``columns``
    maps canonical names to source headers. Multiwell inputs require selection.
    Rank, parent, age and lithology are preserved. Depth metadata and a structured
    import report are stored in ``DataFrame.attrs``. With ``well``, compatible
    length units are converted; incompatible reference/datum joins are rejected.
    ``rank`` (e.g. ``"formation"``) keeps one stratigraphic level of a
    hierarchical table; NPD-style names ("Etive Fm.") carry their rank.
    """
    if hasattr(source, "to_dataframe"):
        source = source.to_dataframe()
    elif isinstance(source, Mapping):
        source = [{"formation_name": name, "top_depth": depth} for name, depth in source.items()]
    elif not isinstance(source, (str, Path, pd.DataFrame)):
        records = []
        for item in source:
            if isinstance(item, Mapping):
                records.append(item)
            elif hasattr(item, "top_depth"):
                records.append({"formation_name": item.name, "top_depth": item.top_depth,
                                "bottom_depth": item.bottom_depth, "lithology": item.lithology,
                                "age": getattr(item, "age", ""),
                                **item.metadata})
            elif hasattr(item, "depth"):
                records.append({"formation_name": item.name, "top_depth": item.depth})
            else:
                records.append({"formation_name": item[0], "top_depth": item[1]})
        source = records
    aliases = {
        "formation_name": ("formationname", "formation", "form", "formationgroup", "name", "surface", "zone", "zonename"),
        "top_depth": ("topdepth", "top", "mdtop", "topmd", "md", "depth", "measureddepth", "from"),
        "bottom_depth": ("bottomdepth", "bottom", "base", "basemd", "mdbottom", "to"),
        "lithology": ("lithology", "majoritylithology"),
        "age": ("age", "geologicalage"), "rank": ("rank", "stratigraphicrank"),
        "parent": ("parent", "parentformation", "parentgroup"),
    }
    table = _prepare_table(source, aliases, ("formation_name", "top_depth"),
                           columns=columns, well_name=well_name, well=well,
                           depth_unit=depth_unit, depth_reference=depth_reference, datum=datum,
                           sheet_name=sheet_name, sep=sep, encoding=encoding)
    table = _validate_depths(table, ("top_depth", "bottom_depth"), well=well)
    for index, row in table.iterrows():
        name = row["formation_name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Geology row {index + 2}: provide a formation name.")
        if "bottom_depth" in table and row["top_depth"] >= row["bottom_depth"]:
            raise ValueError(f"Geology row {index + 2}: top_depth must be less than bottom_depth.")
        table.at[index, "formation_name"] = re.sub(r"\s+top\.?$", "", name.strip(), flags=re.IGNORECASE)
    _infer_rank_from_names(table)
    if rank is not None:
        if "rank" not in table:
            raise ValueError(
                "rank= needs a rank column or NPD-style names ending in Gp./Fm./Mb."
            )
        selected = table["rank"].astype(str).str.strip().str.lower() == str(rank).strip().lower()
        if not selected.any():
            available = ", ".join(repr(str(value)) for value in table["rank"].dropna().unique())
            raise ValueError(f"No formations at rank {rank!r}; available ranks: {available}.")
        attrs = dict(table.attrs)
        table = table.loc[selected].copy()
        table.attrs = attrs
    return table.sort_values("top_depth", kind="stable").reset_index(drop=True)


_RANK_SUFFIXES = (
    ("group", re.compile(r"(?:\bGP\.?|\bGROUP)$", re.IGNORECASE)),
    ("formation", re.compile(r"(?:\bFM\.?|\bFORMATION)$", re.IGNORECASE)),
    ("member", re.compile(r"(?:\bMB\.?|\bMBR\.?|\bMEMBER)$", re.IGNORECASE)),
)


def _infer_rank_from_names(table):
    """Add a ``rank`` column from NPD-style names ("Brent Gp.", "Etive Fm.", ...).

    Only applied when the table has no rank column and every name carries a
    group, formation or member suffix, as in NPD picks.
    """
    if "rank" in table or table.empty:
        return
    ranks = []
    for name in table["formation_name"]:
        rank = next((label for label, pattern in _RANK_SUFFIXES if pattern.search(str(name).strip())), None)
        if rank is None:
            return
        ranks.append(rank)
    table["rank"] = ranks
    found = sorted(set(ranks), key=[label for label, _ in _RANK_SUFFIXES].index)
    table.attrs["import_report"]["diagnostics"].append(
        "Stratigraphic rank inferred from name suffixes (GP./Fm./Mb.): " + ", ".join(found) + "."
    )

