"""GeoLink benchmark context computed from raw LAS.

The dataset profile interprets the input LAS, then exports six category contexts
using the NPD tops workbook and GeoLink lithology codebook. Dataset-specific
well settings and reservoir selections are applied during the calculation.

Each well is interpreted once from its raw LAS. A preparatory step
standardizes the input curves and stores them at the delivered LAS precision
(five decimals), and computes the reservoir flag that selects the SWIRR_APP
samples (RESFLAG) and the bad-hole flag BADHOLEFLAG with the package's
first-run settings; the workflow then runs with the package's context settings. ``processed_las`` holds that
interpretation, so each context statistic is the statistic of that LAS over
the group's depth window, and each ``workflow_json`` trace records the
preparatory step, the workflow steps and the final QC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from petrophys.reporting._single_run import _historical_standardization

#: Wells of the GeoLink package, by file stem.
GEOLINK_WELLS = (
    "15_9-13", "15_9-15", "15_9-17", "15_9-18", "15_9-2", "15_9-23", "15_9-4", "15_9-5",
    "15_9-7", "16_10-4", "16_2-11 A", "16_2-6", "25_10-10", "25_10-8", "25_10-9", "25_11-19 S",
    "30_6-14", "30_6-5", "31_4-4", "34_10-21", "34_10-23", "34_10-35", "34_11-2 S", "35_11-11",
    "35_11-6", "35_11-7", "35_4-1", "35_8-1", "35_8-2", "35_9-2", "35_9-7", "35_9-9",
)


def _first_run_flags(file_path: Path, well) -> dict:
    """Prepare a raw GeoLink well: inputs and reservoir flags of the package's first-run settings.

    See :func:`petrophys.reporting._single_run.prepare_from_first_run`.
    """
    from petrophys.reporting._single_run import RESERVOIR_FLAGS, prepare_from_first_run
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.workflow.zoned_batch import _interpret_las_file, _serialize_step_result_with_well

    result, _ = _interpret_las_file(Path(file_path), DynamicWorkflowConfig(calculation_profile="geolink"))
    return prepare_from_first_run(
        well,
        result.well,
        export_options={
            "depth_unit_label": getattr(result.well, "_source_depth_unit_label", None),
            "pad_header_by_bytes": True,
        },
        reader_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
        used_curves=RESERVOIR_FLAGS,
        calculation_steps=[
            _serialize_step_result_with_well(step, result.well) for step in result.base_result.steps_executed
        ],
    )


@_historical_standardization
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
        Folder for ``processed_las``: the LAS the context is computed from, with
        its ``workflow_json`` traces.
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

    if not any(Path(input_dir).glob(pattern)):
        raise FileNotFoundError(f"No GeoLink LAS files matching {pattern!r} in {input_dir}")
    results = export_reference_json_work_package_by_category_formation_lithology(
        input_dir=input_dir,
        output_dir=output_dir,
        tops_file=tops_file,
        pattern=pattern,
        well_names=well_names,
        lithology_mapping_file=lithology_mapping_file,
        # SWIRR_APP is picked from the reservoir flag of the preparatory step.
        workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
        raw_input_dir=input_dir,
        processed_las_dir=Path(work_dir) / "processed_las",
        prepare_input=_first_run_flags,
    )
    failed = [item for item in results if not item.success]
    if failed:
        raise RuntimeError(
            "GeoLink context failed: "
            + "; ".join(f"{Path(item.source_file).stem}: {' | '.join(item.errors)}" for item in failed[:5])
        )
    write_reference_curve_name_mapping(output_dir)
    return results
