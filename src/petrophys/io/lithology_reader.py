"""Read label-preserving categorical lithology intervals."""

import math
import warnings

import pandas as pd

from petrophys.io.geology_reader import _key, _prepare_table, _read_table, _validate_depths
from petrophys.exceptions import DataQualityWarning


#: Codebook headers (compared without case, spaces or punctuation) of the code
#: column and of the rock-name column, in order of preference.
_CODE_COLUMNS = ("code", "abbreviation", "lithologycode", "lithcode", "faciescode", "codes", "id", "value")
_LABEL_COLUMNS = (
    "lithology", "attribute", "lithologyattribute", "rock", "rocktype", "rockname", "lith",
    "lithologyname", "facies", "faciesname", "name", "label", "description",
)


def read_lithology_codes(source, *, code_column=None, label_column=None, sheet_name=None, sep=None):
    """Read a codebook without confusing rock names with grouping columns.

    CSV/TSV/Excel/DataFrames are supported. The code column is found under
    ``code``, ``abbreviation``, ``lithology code``, ``lith code``,
    ``facies code``, ``id`` or ``value``; the rock name under ``lithology``,
    ``attribute``, ``lithology attribute``, ``rock``, ``rock type``,
    ``rock name``, ``lith``, ``facies``, ``name``, ``label`` or
    ``description`` (case, spaces and punctuation ignored), so Kingdom
    ``Abbreviation,Attribute`` tables read as they are. A table of two other
    columns is read as code, then name. ``Attribute`` takes precedence over
    ``Lithology Attribute``; pass label_column explicitly to select a
    different grouping. Empty labels and duplicate codes are errors.
    """
    table = _read_table(source, sheet_name=sheet_name, sep=sep)
    columns = {_key(column): column for column in table.columns}
    if len(columns) != len(table.columns):
        raise ValueError("Duplicate lithology codebook column names")
    code = code_column if code_column is not None else next(
        (columns[name] for name in _CODE_COLUMNS if name in columns), None)
    label = label_column if label_column is not None else next(
        (columns[name] for name in _LABEL_COLUMNS if name in columns and columns[name] != code), None)
    if code_column is None and label_column is None and (code is None or label is None) and len(table.columns) == 2:
        # Two columns under other names: the code, then the rock name.
        code, label = table.columns[0], table.columns[1]
    if code not in table or label not in table:
        raise ValueError(
            "Lithology codebook needs a code column (code, abbreviation, lithology code, id, ...) and a "
            "rock-name column (lithology, attribute, rock type, name, ...); found "
            f"{', '.join(map(str, table.columns))}. Pass code_column and label_column, or a two-column table."
        )
    if table.empty:
        raise ValueError("Lithology codebook contains no rows")
    if label_column is None and "attribute" in columns and "lithologyattribute" in columns:
        if not table[columns["attribute"]].astype(str).equals(table[columns["lithologyattribute"]].astype(str)):
            warnings.warn(
                f"Lithology codebook contains different Attribute and Lithology Attribute labels; "
                f"using {label!r}. Pass label_column explicitly to choose another grouping.",
                DataQualityWarning, stacklevel=2,
            )
    result = {}
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        value, text = row[code], row[label]
        if pd.isna(value) or pd.isna(text) or not str(value).strip() or not str(text).strip():
            raise ValueError(f"Lithology codebook row {row_number}: code and label must be nonempty")
        value = str(value).strip()
        try:
            number = float(value)
        except ValueError:
            pass
        else:
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"Lithology codebook row {row_number}: numeric codes must be finite integers")
            value = int(number)
        if value in result:
            raise ValueError(f"Duplicate lithology code {value!r}")
        result[value] = str(text).strip()
    return result


def read_lithology_intervals(source, *, columns=None, well_name=None, well=None,
                             depth_unit=None, depth_reference=None, datum=None,
                             sheet_name=None, sep=None, encoding="utf-8-sig") -> pd.DataFrame:
    """Read validated CSV/TSV/Excel/DataFrame intervals, preserving rock labels.

    Required columns are top, bottom and lithology. Optional well/depth metadata
    use the same contract as read_formations. Columns maps canonical names to
    source headers; sheet_name and sep support vendor tables without guessing.
    Gaps remain gaps, while overlaps and invalid rows raise actionable errors.
    Unknown rock names are retained for display, never assigned a routing family.
    """
    aliases = {
        "top": ("top", "topdepth", "from"),
        "bottom": ("bottom", "bottomdepth", "base", "to"),
        "lithology": ("lithology", "lith", "rocktype", "rock", "litho"),
    }
    table = _prepare_table(
        source, aliases, ("top", "bottom", "lithology"), columns=columns,
        well_name=well_name, well=well, depth_unit=depth_unit,
        depth_reference=depth_reference, datum=datum, sheet_name=sheet_name,
        sep=sep, encoding=encoding, strict_columns=columns is None,
    )
    table = _validate_depths(table, ("top", "bottom"), well=well)
    for index, row in table.iterrows():
        if row["top"] >= row["bottom"]:
            raise ValueError(f"Lithology row {index + 2}: top must be less than bottom.")
        label = row["lithology"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Lithology row {index + 2}: provide a nonempty lithology name.")
        table.at[index, "lithology"] = label.strip()
    table = table.sort_values("top", kind="stable")
    previous_bottom = None
    for index, row in table.iterrows():
        if previous_bottom is not None and row["top"] < previous_bottom:
            raise ValueError(
                f"Lithology row {index + 2}: overlapping intervals; provide one "
                "categorical lithology per depth. Mixtures require separate data."
            )
        previous_bottom = row["bottom"]
    return table.reset_index(drop=True)
