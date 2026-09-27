"""Read label-preserving categorical lithology intervals."""

import pandas as pd

from petrophys.io.geology_reader import _prepare_table, _validate_depths


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
