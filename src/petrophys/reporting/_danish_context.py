"""The Danish eval context package, computed from its raw LAS files.

The package was made in three steps, each reproduced here:

1. Preparation. Each raw LAS gets the depth in feet (``DEPT``), the shallow
   resistivity under its standard name (RSHAL becomes RSHA, written last), the
   standard unit label of every standard curve (NPHI labelled PU when its
   values are in percent), and, when it has none, a ``LITHOLOGY_GEOLINK`` code
   curve from each formation's majority lithology in the tops workbook.
2. Interpretation of the prepared LAS with the GeoLink re-run's calculations
   (:func:`petrophys.workflow.dynamic_workflow._dataset_calculations`).
3. Context, one lithology group per formation of the tops workbook, named by
   its majority lithology (:func:`danish_group_layout`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Wells of the Danish package, by file stem.
DANISH_WELLS = ("east-rosa-flank-1", "elin-1", "elna-1", "ida-1", "skjold-flank-1")

_M_PER_FT = 0.3048
#: Feet per metre in the package's formation depths: metre wells' depths were
#: converted with 3.28084, the prepared LAS with 1/0.3048.
_LAYOUT_FT_PER_M = 3.28084
#: GeoLink codes the preparation gave each majority lithology of the tops.
MAJORITY_LITHOLOGY_CODES = {
    "sandstone": 1,
    "siltstone": 5,
    "claystone": 8,
    "chalk": 9,
    "limestone": 11,
    "marl": 13,
}
_LITHOLOGY_CODE_DESCRIPTION = "Lithology GeoLink Integer Code from Danish formation majority lithology"
#: Curves the preparation renamed to their standard name.
_PREPARATION_RENAMES = {"RSHAL": "RSHA"}
_TOPS_COLUMNS = ("Well", "Formation", "MD_Top", "MD_Bottom", "MajorityLithology")


def read_danish_tops(tops_file: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Formation rows of the Danish tops workbook, by lower-case well name.

    The workbook has a row per well and formation with ``Well``,
    ``Formation``, ``MD_Top``, ``MD_Bottom`` (feet, ``DepthUnit``) and
    ``MajorityLithology``.
    """
    import pandas as pd

    frame = pd.read_excel(tops_file)
    columns = {str(column).strip(): column for column in frame.columns}
    missing = [name for name in _TOPS_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"Danish tops workbook {tops_file} lacks columns: {', '.join(missing)}")
    if "DepthUnit" in columns:
        units = {str(value).strip().lower() for value in frame[columns["DepthUnit"]].dropna()}
        if units - {"feet", "ft", "foot"}:
            raise ValueError(f"Danish tops depths must be in feet, got {sorted(units)}")
    tops: dict[str, list[dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        well = row[columns["Well"]]
        if not isinstance(well, str) or not well.strip():
            continue
        tops.setdefault(well.strip().lower(), []).append(
            {
                "formation": str(row[columns["Formation"]]).strip(),
                "top": float(row[columns["MD_Top"]]),
                "bottom": float(row[columns["MD_Bottom"]]),
                "lithology": str(row[columns["MajorityLithology"]]).strip(),
            }
        )
    return tops


def prepare_danish_las(raw_file: str | Path, tops: Sequence[Mapping[str, Any]], output_file: str | Path) -> Path:
    """Write the prepared LAS the Danish package interpreted (see the module docstring)."""
    from petrophys.io import export_las, read_las
    from petrophys.preprocessing.standardization.alias_map import STANDARD_UNITS

    well = read_las(raw_file)
    if well.depth_unit == "m":
        depth_ft = np.asarray(well.depth, dtype=float) / _M_PER_FT
        well.depth = depth_ft
        well.depth_unit = "ft"
        for mnemonic in well.curves:
            well.get(mnemonic).depth = depth_ft
    if well.get("LITHOLOGY_GEOLINK") is None:
        depth = np.asarray(well.depth, dtype=float)
        codes = np.full(well.n_samples, np.nan)
        for formation in tops:
            lithology = formation["lithology"].lower()
            if lithology not in MAJORITY_LITHOLOGY_CODES:
                raise ValueError(f"No GeoLink code for majority lithology {formation['lithology']!r}")
            inside = (depth >= formation["top"]) & (depth < formation["bottom"])
            codes[inside] = MAJORITY_LITHOLOGY_CODES[lithology]
        well.add_curve("LITHOLOGY_GEOLINK", codes, "", _LITHOLOGY_CODE_DESCRIPTION)
    for old, new in _PREPARATION_RENAMES.items():
        if well.get(old) is not None and well.get(new) is None:
            well.rename_curve(old, new)
    for mnemonic in well.curves:
        unit = STANDARD_UNITS.get(mnemonic)
        if unit is None:
            continue
        curve = well.get(mnemonic)
        if mnemonic == "NPHI":
            values = np.asarray(curve.data, dtype=float)
            finite = values[np.isfinite(values)]
            unit = "PU" if finite.size and float(np.median(finite)) > 1.0 else "v/v"
        curve.unit = unit
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_las(well, output, depth_unit_label="FT")
    return output


def danish_group_layout(raw_file: str | Path, tops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One context group per formation of ``tops``, as the Danish package laid them out.

    A formation spans the raw samples from its top to its bottom, inclusive,
    with metre depths in feet by 3.28084; its group, named after the majority
    lithology (``"CHALK_GROUP_1"``), spans the same samples with both ends
    rounded to 0.01 ft, and the formation's thickness is rounded alike.
    """
    from petrophys.io import read_las

    well = read_las(raw_file)
    depth = np.asarray(well.depth, dtype=float)
    if well.depth_unit == "m":
        depth = depth * _LAYOUT_FT_PER_M
    groups = []
    for formation in tops:
        inside = (depth >= formation["top"]) & (depth <= formation["bottom"])
        if not inside.any():
            continue
        top, bottom = float(depth[inside][0]), float(depth[inside][-1])
        lithology = formation["lithology"].upper()
        groups.append(
            {
                "formation_name": formation["formation"],
                "formation_depth_info": {
                    "min_depth": top,
                    "max_depth": bottom,
                    "thickness": round(bottom - top, 2),
                },
                "group_name": f"{lithology}_GROUP_1",
                "lithology": lithology,
                "start_depth": round(top, 2),
                "end_depth": round(bottom, 2),
            }
        )
    return groups


def reproduce_danish_context(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tops_file: str | Path,
    lithology_mapping_file: str | Path,
    work_dir: str | Path,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
    recursive: bool = False,
):
    """Compute the Danish context package from its raw LAS files.

    Parameters
    ----------
    input_dir : str or Path
        Folder of the Danish raw LAS files.
    output_dir : str or Path
        Folder for the category context JSON.
    tops_file : str or Path
        The Danish tops workbook (``Danish_formation_tops_for_context.xlsx``).
    lithology_mapping_file : str or Path
        The GeoLink/Kingdom lithology codebook the GeoLink package used.
    work_dir : str or Path
        Folder for the prepared LAS (``prepared_las``) and the interpreted LAS
        with their workflow traces (``processed_las``).
    well_names : sequence of str, optional
        Only these wells (file stems).

    Returns
    -------
    list of ExampleFormatExportResult
        One per context file.
    """
    from petrophys.reporting.reference_export import (
        _filter_input_files_by_well_names,
        export_reference_json_work_package_by_category_formation_lithology,
        write_reference_curve_name_mapping,
    )
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, _dataset_calculations
    from petrophys.workflow.zoned_batch import process_las_directory_with_workflow

    source = Path(input_dir)
    work = Path(work_dir)
    prepared_dir = work / "prepared_las"
    processed_dir = work / "processed_las"
    files = sorted(source.rglob(pattern) if recursive else source.glob(pattern))
    files = _filter_input_files_by_well_names(files, well_names)
    if not files:
        raise FileNotFoundError(f"No Danish LAS files matching {pattern!r} in {source}")
    tops = read_danish_tops(tops_file)
    layout = {}
    for raw_file in files:
        well_tops = tops.get(raw_file.stem.lower(), [])
        prepare_danish_las(raw_file, well_tops, prepared_dir / f"{raw_file.stem}.las")
        layout[raw_file.stem] = danish_group_layout(raw_file, well_tops)

    stems = [raw_file.stem for raw_file in files]
    with _dataset_calculations("danish"):
        batch = process_las_directory_with_workflow(
            prepared_dir,
            processed_dir,
            config=DynamicWorkflowConfig(calculation_profile="geolink"),
            well_names=stems,
        )
        failed = [item for item in batch if not item.success]
        if failed:
            raise RuntimeError(
                "Danish interpretation failed: "
                + "; ".join(f"{Path(item.source_file).stem}: {' | '.join(item.errors)}" for item in failed)
            )
        results = export_reference_json_work_package_by_category_formation_lithology(
            input_dir=processed_dir,
            output_dir=output_dir,
            well_names=stems,
            lithology_mapping_file=lithology_mapping_file,
            # The package picked SWIRR_APP from the reservoir flag already on
            # the interpreted LAS, as the GeoLink package did.
            workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
            raw_input_dir=prepared_dir,
            group_layout=layout,
        )
    write_reference_curve_name_mapping(output_dir)
    return results
