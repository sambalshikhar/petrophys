"""GeoLink training context computed from raw LAS.

The training package covers 191 Norwegian wells whose raw LAS files carry the
operators' interpretation parameters in their headers
(``LAS_with_petrophys_params``). Its context was computed from an
interpretation of an interpretation: a first run interpreted each raw LAS and
stored the result as LAS, and the context interpreted that LAS again, both
with the calculations of the "geolink" profile's training-package variant
(:func:`petrophys.workflow.dynamic_workflow._dataset_calculations`).

Each well is interpreted once from its raw LAS. A preparatory step
standardizes the input curves and stores them at the delivered LAS precision
(five decimals), and computes the reservoir flag that selects the SWIRR_APP
samples (RESFLAG) with the first run's settings; the workflow then runs with
the context's settings. ``processed_las`` holds that interpretation after the
context's value-range QC and the recomputation of the curves that depend on
its inputs, so each context statistic is the statistic of that LAS over the
group's samples, and each ``workflow_json`` trace records the preparatory
step, the workflow steps and the final QC.

The context lists the input logs of :func:`original_curves`, and lists no bit
size (BS) or oil saturation (SO): neither appears among the input or derived
logs, and neither gets statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from petrophys.reporting._single_run import _historical_standardization

#: Wells of the GeoLink training package, by file stem.
GEOLINK_TRAIN_WELLS = (
    "15_9-12", "15_9-14", "15_9-19 A", "15_9-19 B", "15_9-22", "15_9-8", "15_9-9", "16_1-2",
    "16_1-6 A", "16_10-1", "16_10-2", "16_10-3", "16_10-5", "16_11-1 S", "16_11-2", "16_2-16",
    "16_2-7", "16_4-1", "16_4-2", "16_5-3", "16_6-1", "16_7-1", "16_7-2", "16_7-3", "16_7-4",
    "16_7-5", "16_7-6", "16_7-7 S", "16_7-8 S", "16_8-1", "16_9-1", "17_11-1", "17_4-1", "24_12-2",
    "25_10-2 R", "25_10-3", "25_10-4 R", "25_10-8 A", "25_11-1", "25_11-15", "25_11-23", "25_11-24",
    "25_11-5", "25_2-13 T4", "25_2-14", "25_2-5", "25_2-6", "25_2-7", "25_3-1", "25_4-1", "25_4-5",
    "25_5-1", "25_5-2", "25_5-3", "25_5-4", "25_6-1", "25_6-2", "25_6-3", "25_7-2", "25_8-1",
    "25_8-5 S", "25_8-7", "25_8-9", "25_9-1", "26_4-1", "29_3-1", "29_6-1", "30_11-3", "30_2-1",
    "30_2-2", "30_3-2 R", "30_3-3", "30_3-4 R", "30_3-5 S", "30_4-1", "30_4-2", "30_6-11",
    "30_6-19", "30_6-22", "30_6-23", "30_6-8", "30_7-7", "30_9-13 S", "31_2-1", "31_2-10",
    "31_2-17 A", "31_2-19 S", "31_2-2 R", "31_2-21 S", "31_2-3", "31_2-7", "31_2-8", "31_2-9",
    "31_3-1", "31_3-2", "31_3-3", "31_3-4", "31_4-10", "31_4-2", "31_4-3", "31_4-5", "31_4-6",
    "31_4-8", "31_5-2 R", "31_5-3", "31_5-4 S", "31_5-5 R", "31_5-6", "31_6-1", "31_6-5", "31_6-8",
    "32_2-1", "33_5-2", "33_6-1", "33_6-2", "33_6-3 S", "33_9-1", "33_9-11", "33_9-17", "33_9-6",
    "34_10-12", "34_10-16 R", "34_10-17", "34_10-18", "34_10-19", "34_10-33", "34_10-5", "34_10-7",
    "34_11-1", "34_11-3 T2", "34_12-1", "34_2-2 R", "34_2-4", "34_3-1 A", "34_3-1 S", "34_3-2 S",
    "34_3-3 A", "34_3-3 S", "34_4-10 R", "34_4-2", "34_4-3", "34_4-5", "34_5-1 A", "34_5-1 S",
    "34_6-1 S", "34_7-12", "34_7-13", "34_7-15 S", "34_7-16", "34_7-20", "34_7-21", "34_7-5",
    "34_8-1", "34_8-3", "34_8-6", "34_8-7", "35_10-1", "35_10-2", "35_11-1", "35_11-10", "35_11-12",
    "35_11-13", "35_11-15 S", "35_11-2", "35_11-5", "35_11-9", "35_12-1", "35_12-4 S", "35_3-1",
    "35_3-2", "35_3-4", "35_3-5", "35_3-7 S", "35_6-2 S", "35_7-1 S", "35_8-3", "35_8-4",
    "35_8-5 S", "35_8-6 S", "35_9-1", "35_9-10 S", "35_9-5", "35_9-6 S", "35_9-8", "36_7-1",
    "36_7-3", "6_3-1", "6_3-2", "7_1-1", "7_1-2 S", "7_3-1",
)


#: Core curves the training package's LAS carry beside the logs: the context
#: does not count them as input logs.
CORE_CURVES = ("PHIC", "GDEN", "KH_CORE", "KV_CORE", "SWC", "SOC")


def original_curves(well) -> set[str]:
    """The curves of a raw training-package well that the context counts as input logs.

    The package's LAS add core data (:data:`CORE_CURVES`) to the logged
    curves, and its context lists no bit size (BS) among the input logs.
    """
    return set(well.curves) - set(CORE_CURVES) - {"BS"}


def _without_bit_size(well) -> bool:
    """True when the well records no bit size: no BS curve and no header bit size."""
    return well.get("BS") is None and getattr(well.header, "bit_size", None) is None


def _add_oil_saturation(well) -> None:
    """Add SO = 1 - SW, as the package's interpretation of a well without a bit size held it."""
    import numpy as np

    sw = well.get("SW")
    if sw is None or well.get("SO") is not None:
        return
    values = np.asarray(sw.data, dtype=float)
    well.add_curve("SO", np.where(np.isfinite(values), 1.0 - values, np.nan), "v/v", "Oil Saturation")


def _first_run_flags(file_path: Path, well) -> dict:
    """Prepare a raw training-package well: inputs and reservoir flag of the first run.

    See :func:`petrophys.reporting._single_run.prepare_from_first_run`; the
    context's input logs are :func:`original_curves`.
    """
    from petrophys.reporting._single_run import RESERVOIR_FLAGS, prepare_from_first_run
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.workflow.zoned_batch import _interpret_las_file, _serialize_step_result_with_well

    raw_curves = original_curves(well)
    result, _ = _interpret_las_file(Path(file_path), DynamicWorkflowConfig(calculation_profile="geolink"))
    preparation = prepare_from_first_run(
        well,
        result.well,
        export_options={"depth_unit_label": getattr(result.well, "_source_depth_unit_label", None)},
        reader_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
        used_curves=[flag for flag in RESERVOIR_FLAGS if flag[0] == "RESFLAG"],
        calculation_steps=[
            _serialize_step_result_with_well(step, result.well) for step in result.base_result.steps_executed
        ],
    )
    preparation["raw_curves"] = raw_curves
    if _without_bit_size(well) and (well.get("CALI") is not None or well.get("DRHO") is not None):
        # The package's wells without a bit size were interpreted with the
        # rule that flags bad hole from |DRHO| alone, by an interpretation
        # that also held the oil saturation SO = 1 - SW.
        curve_order = preparation["after_workflow"]

        def after_workflow(interpreted) -> None:
            curve_order(interpreted)
            _add_oil_saturation(interpreted)

        preparation["after_workflow"] = after_workflow
        preparation["preparatory_steps"].append(
            {
                "step": "oil_saturation",
                "output_curves": ["SO"],
                "method": "SO = 1 - SW, added to the interpreted well after the workflow",
                "description": "The package interpreted wells without a bit size with the oil saturation "
                "beside SW; SO counts among the logs of DATA_COMPLETENESS, and the context gives it no "
                "statistics.",
            }
        )
    return preparation


def _export_geolink_train(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    processed_las_dir: str | Path,
    processed_las_only: bool,
    well_names: Optional[Sequence[str]],
    pattern: str,
    **options,
):
    """Interpret the training package's raw LAS with its calculations; see the module docstring."""
    from petrophys.reporting.reference_export import export_reference_json_work_package_by_category_formation_lithology
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, _dataset_calculations

    if not any(Path(input_dir).glob(pattern)):
        raise FileNotFoundError(f"No GeoLink training LAS files matching {pattern!r} in {input_dir}")
    with _dataset_calculations("geolink_train"):
        results = export_reference_json_work_package_by_category_formation_lithology(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            well_names=well_names,
            # SWIRR_APP is picked from the reservoir flag of the preparatory step.
            workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True),
            raw_input_dir=input_dir,
            processed_las_dir=processed_las_dir,
            prepare_input=_first_run_flags,
            processed_las_only=processed_las_only,
            **options,
        )
    failed = [item for item in results if not item.success]
    if failed:
        raise RuntimeError(
            "GeoLink training context failed: "
            + "; ".join(f"{Path(item.source_file).stem}: {' | '.join(item.errors)}" for item in failed[:5])
        )
    return results


@_historical_standardization
def reproduce_geolink_train_context(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tops_file: str | Path,
    lithology_mapping_file: str | Path,
    work_dir: str | Path,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
):
    """Compute the GeoLink training context package from its raw LAS files.

    Parameters
    ----------
    input_dir : str or Path
        Folder of the package's raw LAS files (``LAS_with_petrophys_params``).
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
    from petrophys.reporting.reference_export import write_reference_curve_name_mapping

    results = _export_geolink_train(
        input_dir,
        output_dir,
        processed_las_dir=Path(work_dir) / "processed_las",
        processed_las_only=False,
        well_names=well_names,
        pattern=pattern,
        tops_file=tops_file,
        lithology_mapping_file=lithology_mapping_file,
    )
    write_reference_curve_name_mapping(output_dir)
    return results


@_historical_standardization
def process_geolink_train_las(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tops_file: Optional[str | Path] = None,
    lithology_mapping_file: Optional[str | Path] = None,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
):
    """Write the processed LAS of the GeoLink training package from its raw LAS files.

    Each raw LAS is interpreted once, as :func:`reproduce_geolink_train_context`
    interprets it, and saved with its trace; no context JSON is written. The
    statistics of each LAS over a context group's samples are that group's
    context statistics.

    Parameters
    ----------
    input_dir : str or Path
        Folder of the package's raw LAS files (``LAS_with_petrophys_params``).
    output_dir : str or Path
        Folder for the processed LAS, with their traces in ``workflow_json``.
    tops_file, lithology_mapping_file : str or Path, optional
        The context's tops workbook and lithology codebook, read and checked
        as the context run reads them. They only group the context, so the
        LAS are the same without them.
    well_names : sequence of str, optional
        Only these wells (file stems).

    Returns
    -------
    list of ExampleFormatExportResult
        One per well; ``output_file`` is its processed LAS.
    """
    return _export_geolink_train(
        input_dir,
        output_dir,
        processed_las_dir=output_dir,
        processed_las_only=True,
        well_names=well_names,
        pattern=pattern,
        tops_file=tops_file,
        lithology_mapping_file=lithology_mapping_file,
    )
