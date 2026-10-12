"""
Regenerate the GeoLink eval context benchmark package.

Runs the category-first context exporter, which reproduces the calculations,
layout, and notes of
context_json_category_all_wells_fm_gp_mb_core_no_metrics_phiecc_sat_latest
from the processed LAS files (eval_processed), the raw LAS files next to them
(eval), the NPD formation tops workbook, and the lithology codebook.

With ``--from-raw`` the input folder holds the raw LAS files instead, and the
script runs the whole package from them: first the interpreted ("processed")
LAS files, then the context from those. Both outputs land under the output
folder.

The archived package mixes two interpretation runs: 30 wells were interpreted
in April, and 15_9-17 and 34_10-23 ten days later with changed calculations,
which also wrote the context. The calculation profiles "benchmark" and
"benchmark_rerun" keep those two sets of calculations fixed while the library's
default calculations change. ``--las-profile`` sets the profile for every well
and ``--well-profile`` names the exceptions, so one command can reproduce the
whole package:

    python scripts/generate_benchmark_context.py --from-raw \
      --input-dir PATH/eval --output-dir NEW_DIR \
      --well-profile 15_9-17 benchmark_rerun --well-profile 34_10-23 benchmark_rerun

ELECTROFACIES labels come from k-means clustering, which was unseeded when the
benchmark was made, so they cannot be recovered. The script fixes the string-hash seed, so
reruns write byte-identical files.

petrophys-context and petrophys.export_context run the same calculations; this
script adds the eval-package defaults (raw LAS folder, NPD tops and codebook).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from petrophys.reporting.reference_export import (  # noqa: E402
    export_reference_json_work_package_by_category_formation_lithology,
    write_reference_curve_name_mapping,
)
from petrophys.workflow import DynamicWorkflowConfig  # noqa: E402
from petrophys.workflow.zoned_batch import (  # noqa: E402
    process_las_directory_with_workflow,
)


_PROFILES = ("benchmark", "benchmark_rerun", "current")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the GeoLink eval context benchmark: category-first JSON by well, "
            "formation, and lithology group."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the processed LAS files (eval_processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the generated package.",
    )
    parser.add_argument(
        "--tops-file",
        type=Path,
        default=None,
        help="Excel workbook containing formation tops. Defaults to "
             "'NPD stratigraphic picks north sea.xlsx' next to --input-dir.",
    )
    parser.add_argument(
        "--lithology-mapping-file",
        type=Path,
        default=None,
        help="Workbook used to map LITHOLOGY_GEOLINK codes to lithology-group labels. "
             "Defaults to 'Litho_Nomenclature_Kingdom.xlsx' next to --input-dir.",
    )
    parser.add_argument(
        "--raw-input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the raw LAS files, used to decide which curves are inputs. "
            "If omitted and --input-dir ends with _processed, the sibling raw directory is used."
        ),
    )
    parser.add_argument(
        "--well",
        dest="well_names",
        action="append",
        default=[],
        help="Limit processing to a specific well name or LAS stem. Repeat for multiple wells.",
    )
    parser.add_argument(
        "--las-profile",
        choices=_PROFILES,
        default="benchmark",
        help=(
            "Calculation profile for the --from-raw interpretation step. The "
            "archived package mixes two runs: 30 wells need 'benchmark', while "
            "15_9-17 and 34_10-23 were reprocessed later and need 'benchmark_rerun'."
        ),
    )
    parser.add_argument(
        "--context-profile",
        choices=_PROFILES,
        default="benchmark_rerun",
        help=(
            "Calculation profile for the context step. The package's context was "
            "written with the 'benchmark_rerun' calculations."
        ),
    )
    parser.add_argument(
        "--well-profile",
        nargs=2,
        action="append",
        metavar=("WELL", "PROFILE"),
        dest="well_profiles",
        default=[],
        help=(
            "Interpret one well with a different profile than --las-profile. "
            "Repeat per well, e.g. --well-profile 15_9-17 benchmark_rerun."
        ),
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Treat --input-dir as raw LAS files: interpret them into "
            "<output-dir>/processed_las first, then build the context from those."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.las",
        help="Glob pattern used to select LAS files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for LAS files recursively under --input-dir.",
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=2.0,
        help="Minimum lithology-group thickness to export, in the well depth unit.",
    )
    return parser


def _normalize_well(name: str) -> str:
    return str(name).strip().lower().replace("/", "_")


def _wells_by_profile(args) -> dict:
    """Group the selected wells by the profile that should interpret them."""
    overrides = {_normalize_well(well): profile for well, profile in args.well_profiles}
    unknown = sorted(set(overrides.values()) - set(_PROFILES))
    if unknown:
        raise SystemExit(
            f"--well-profile accepts {', '.join(_PROFILES)}; got {', '.join(unknown)}"
        )
    pattern = args.pattern
    files = sorted(
        args.input_dir.rglob(pattern) if args.recursive else args.input_dir.glob(pattern)
    )
    wanted = {_normalize_well(name) for name in (args.well_names or [])}
    groups: dict = {}
    for item in files:
        stem = _normalize_well(item.stem)
        if wanted and stem not in wanted:
            continue
        groups.setdefault(overrides.get(stem, args.las_profile), []).append(item.stem)
    missing = sorted(set(overrides) - {_normalize_well(s) for g in groups.values() for s in g})
    if missing:
        raise SystemExit(f"--well-profile names wells not in {args.input_dir}: {', '.join(missing)}")
    return groups


def _rerun_with_fixed_hash_seed() -> None:
    """Re-run under PYTHONHASHSEED=0 so set iteration, and therefore list order, is stable."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.tops_file is None:
        args.tops_file = args.input_dir.parent / "NPD stratigraphic picks north sea.xlsx"
    if args.lithology_mapping_file is None:
        args.lithology_mapping_file = args.input_dir.parent / "Litho_Nomenclature_Kingdom.xlsx"

    raw_input_dir = args.raw_input_dir
    context_input_dir = args.input_dir
    processed_dir = None
    if args.from_raw:
        processed_dir = Path(args.output_dir) / "processed_las"
        batch = []
        for profile, wells in _wells_by_profile(args).items():
            batch.extend(
                process_las_directory_with_workflow(
                    input_dir=args.input_dir,
                    output_dir=processed_dir,
                    pattern=args.pattern,
                    recursive=args.recursive,
                    config=DynamicWorkflowConfig(calculation_profile=profile),
                    well_names=wells,
                )
            )
            print(f"Interpreted LAS: {processed_dir}  ({profile}: {len(wells)} well(s))")
        failed_wells = [item.source_file for item in batch if not item.success]
        print(f"  wells: {len(batch)}  written: {len(batch) - len(failed_wells)}")
        for source in failed_wells:
            print(f"  FAILED {source}")
        # The context classifies input logs against the raw files, and reads the
        # interpreted LAS written above.
        raw_input_dir = raw_input_dir or args.input_dir
        context_input_dir = processed_dir

    results = export_reference_json_work_package_by_category_formation_lithology(
        # The package picked SWIRR_APP from the reservoir flag already on the
        # interpreted LAS, which is no longer the default.
        workflow_config=DynamicWorkflowConfig(
            calculation_profile=args.context_profile, reuse_input_resflag=True
        ),
        input_dir=context_input_dir,
        output_dir=args.output_dir,
        tops_file=args.tops_file,
        pattern=args.pattern,
        recursive=args.recursive,
        minimum_group_thickness=args.min_thickness,
        well_names=args.well_names,
        lithology_mapping_file=args.lithology_mapping_file,
        raw_input_dir=raw_input_dir,
    )
    mapping_path = write_reference_curve_name_mapping(args.output_dir)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    processed_wells = sorted({Path(item.source_file).stem for item in results if item.source_file})

    if processed_dir is not None:
        print(f"Raw LAS:        {args.input_dir}")
    print(f"Input LAS:      {context_input_dir}")
    print(f"Tops file:      {args.tops_file}")
    print(f"Output root:    {args.output_dir}")
    print(f"Pattern:        {args.pattern}")
    print(f"Recursive:      {args.recursive}")
    print(f"Min thickness:  {args.min_thickness}")
    print(f"Litho map:      {args.lithology_mapping_file}")
    print(f"Raw input LAS:  {raw_input_dir or 'auto'}")
    if args.well_names:
        print(f"Wells filter:   {', '.join(args.well_names)}")
    print(f"Mapping:        {mapping_path}")
    print(f"Wells:          {len(processed_wells)}")
    print(f"JSON files:     {total}")
    print(f"Success:        {successful}")
    print(f"Failed:         {failed}")

    if failed:
        print("\nFailures:")
        for item in results:
            if item.success:
                continue
            name = Path(item.source_file).stem if item.source_file else "UNKNOWN"
            print(f"- {name}: {' | '.join(item.errors) if item.errors else 'Unknown error'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    _rerun_with_fixed_hash_seed()
    raise SystemExit(main())
