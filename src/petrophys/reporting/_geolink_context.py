"""The GeoLink eval context package, computed from its raw LAS files.

The package was made in two steps, both reproduced here with
``calculation_profile="geolink"``:

1. Interpretation of the raw LAS by the zoned batch runner: 30 wells with the
   calculations of the package's main run, 15/9-17 and 34/10-23 with those of
   its re-run.
2. Context from the interpreted LAS, the NPD formation tops workbook and the
   GeoLink/Kingdom lithology codebook, every well with the re-run's
   calculations; the reservoir flag already on the interpreted LAS picks the
   SWIRR_APP samples, as in the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

#: Wells of the GeoLink package, by file stem.
GEOLINK_WELLS = (
    "15_9-13", "15_9-15", "15_9-17", "15_9-18", "15_9-2", "15_9-23", "15_9-4", "15_9-5",
    "15_9-7", "16_10-4", "16_2-11 A", "16_2-6", "25_10-10", "25_10-8", "25_10-9", "25_11-19 S",
    "30_6-14", "30_6-5", "31_4-4", "34_10-21", "34_10-23", "34_10-35", "34_11-2 S", "35_11-11",
    "35_11-6", "35_11-7", "35_4-1", "35_8-1", "35_8-2", "35_9-2", "35_9-7", "35_9-9",
)


def reproduce_geolink_context(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tops_file: str | Path,
    lithology_mapping_file: str | Path,
    work_dir: str | Path,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
):
    """Compute the GeoLink context package from its raw LAS files.

    Parameters
    ----------
    input_dir : str or Path
        Folder of the GeoLink raw LAS files (the package's ``eval`` folder).
    output_dir : str or Path
        Folder for the category context JSON.
    tops_file : str or Path
        The NPD formation tops workbook (``NPD stratigraphic picks north sea.xlsx``).
    lithology_mapping_file : str or Path
        The GeoLink/Kingdom lithology codebook (``Litho_Nomenclature_Kingdom.xlsx``).
    work_dir : str or Path
        Folder for the interpreted LAS and their workflow traces (``processed_las``).
    well_names : sequence of str, optional
        Only these wells (file stems).

    Returns
    -------
    list of ExampleFormatExportResult
        One per context file.
    """
    from petrophys.reporting.reference_export import (
        export_reference_json_work_package_by_category_formation_lithology,
        write_reference_curve_name_mapping,
    )
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.workflow.zoned_batch import process_las_directory_with_workflow

    processed_dir = Path(work_dir) / "processed_las"
    batch = process_las_directory_with_workflow(
        input_dir=input_dir,
        output_dir=processed_dir,
        pattern=pattern,
        config=DynamicWorkflowConfig(calculation_profile="geolink"),
        well_names=well_names,
    )
    if not batch:
        raise FileNotFoundError(f"No GeoLink LAS files matching {pattern!r} in {input_dir}")
    failed = [item for item in batch if not item.success]
    if failed:
        raise RuntimeError(
            "GeoLink interpretation failed: "
            + "; ".join(f"{Path(item.source_file).stem}: {' | '.join(item.errors)}" for item in failed)
        )
    results = export_reference_json_work_package_by_category_formation_lithology(
        input_dir=processed_dir,
        output_dir=output_dir,
        tops_file=tops_file,
        pattern=pattern,
        well_names=[Path(item.source_file).stem for item in batch],
        lithology_mapping_file=lithology_mapping_file,
        # The package picked SWIRR_APP from the reservoir flag already on the
        # interpreted LAS; the library default recomputes the flag.
        workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
        raw_input_dir=input_dir,
    )
    write_reference_curve_name_mapping(output_dir)
    return results
