"""Generate dataset-specific benchmark context for GeoLink, COSTA or Danish.

Pass one LAS file or a folder and select a dataset, or let the script detect
it from well names. Dataset profiles supply the interpretation parameters,
calibration and grouping rules. Raw GeoLink folders require --from-raw;
GeoLink and Danish also require their tops workbook and lithology codebook.
<output-dir>/processed_las holds the LAS (and workflow traces) the context is
computed from, so each context statistic is the statistic of that LAS over the
group's depth window. A --calculation-profile current run from raw LAS keeps
its first interpretation under <output-dir>/first_pass_las.

See docs/benchmark_context.md for complete commands and required inputs.

Example: python scripts/generate_benchmark_context.py --from-raw --input-dir PATH/eval --output-dir NEW_DIR
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
from petrophys.reporting.context_profiles import (  # noqa: E402
    export_context_profile,
    identify_context_profile,
)
from petrophys.api import read  # noqa: E402
from petrophys.workflow import DynamicWorkflowConfig  # noqa: E402
from petrophys.workflow.zoned_batch import (  # noqa: E402
    process_las_directory_with_workflow,
)


#: The command that reproduces the whole package from the raw LAS files.
REPRODUCE_COMMAND = (
    "python scripts/generate_benchmark_context.py --from-raw "
    "--input-dir PATH/eval --output-dir NEW_DIR"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce category-first context JSON for GeoLink, COSTA, or Danish "
            "eval wells. The collection is detected automatically."
        ),
        epilog=f"Reproduce the whole package from the raw LAS files:\n  {REPRODUCE_COMMAND}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="One eval LAS file, or a directory containing eval LAS files.",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "geolink", "costa", "danish"),
        default="auto",
        help="Eval collection (default: detect from the well names).",
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
             "'NPD stratigraphic picks north sea.xlsx' next to --input-dir (GeoLink) or "
             "'Danish_formation_tops_for_context.xlsx' next to it or its parent (Danish).",
    )
    parser.add_argument(
        "--lithology-mapping-file",
        type=Path,
        default=None,
        help="Workbook used to map LITHOLOGY_GEOLINK codes to lithology-group labels. "
             "Defaults to 'Litho_Nomenclature_Kingdom.xlsx' next to --input-dir; "
             "required for Danish.",
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
        "--calculation-profile",
        choices=("geolink", "current"),
        default="geolink",
        help=(
            "Interpretation configuration for GeoLink runs: geolink (default) "
            "or current. COSTA and Danish use their dataset configurations."
        ),
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Treat --input-dir as raw LAS files: interpret them into "
            "<output-dir>/first_pass_las first, then build the context from those."
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


def _rerun_with_fixed_hash_seed() -> None:
    """Re-run under PYTHONHASHSEED=0 so set iteration, and therefore list order, is stable."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))


def _well_key(value: str) -> str:
    return Path(str(value)).stem.strip().casefold().replace("/", "_")


def _input_scope(
    parser: argparse.ArgumentParser,
    input_path: Path,
    well_names: list[str],
) -> tuple[Path, list[str], bool]:
    if not input_path.exists():
        parser.error(f"input LAS path does not exist: {input_path}")
    if input_path.is_dir():
        return input_path, well_names, False
    if input_path.suffix.casefold() != ".las":
        parser.error(f"single-file input must be LAS: {input_path}")
    inferred = input_path.stem
    if well_names and _well_key(inferred) not in {_well_key(name) for name in well_names}:
        parser.error(f"--well does not select input file {input_path.name}")
    return input_path.parent, well_names or [inferred], True


def _detect_dataset(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_path: Path,
    input_dir: Path,
    well_names: list[str],
) -> str:
    if args.dataset != "auto":
        return args.dataset
    profile = identify_context_profile(
        input_path,
        well_names=well_names,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    if profile is not None:
        return profile.name.removesuffix("_v1")
    parser.error(
        "could not identify the eval collection. Use a recognized GeoLink, COSTA, "
        "or Danish LAS file, or set --dataset explicitly."
    )
    raise AssertionError("argparse.error did not exit")


def _looks_like_processed_geolink(path: Path) -> bool:
    try:
        well = read(path)
    except Exception:
        return False
    derived = {"VCL", "PHIE", "SW", "PERM", "NET_PAY", "RESFLAG"}
    return len(derived.intersection(well.curves)) >= 3


def _danish_tops_file(input_dir: Path) -> Path | None:
    for folder in (input_dir.parent, input_dir.parent.parent):
        candidate = folder / "Danish_formation_tops_for_context.xlsx"
        if candidate.exists():
            return candidate
    return None


def _export_profile(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    dataset: str,
    input_dir: Path,
    well_names: list[str],
) -> int:
    if args.calculation_profile != "geolink":
        parser.error(f"the {dataset} profile fixes its interpretation configuration; "
                     "omit the --calculation-profile override")
    if args.min_thickness != 2.0 or args.raw_input_dir is not None:
        parser.error(f"the {dataset} package fixes its grouping and inputs; "
                     "drop --min-thickness and --raw-input-dir")
    tops_file = lithology_mapping_file = None
    if dataset == "geolink":
        tops_file = args.tops_file or input_dir.parent / "NPD stratigraphic picks north sea.xlsx"
        lithology_mapping_file = args.lithology_mapping_file or input_dir.parent / "Litho_Nomenclature_Kingdom.xlsx"
    elif dataset == "danish":
        tops_file = args.tops_file or _danish_tops_file(input_dir)
        lithology_mapping_file = args.lithology_mapping_file
        if tops_file is None or lithology_mapping_file is None:
            parser.error("danish needs --tops-file (its tops workbook) and --lithology-mapping-file "
                         "(the GeoLink codebook, Litho_Nomenclature_Kingdom.xlsx)")
    profile = f"{dataset}_v1"
    results = export_context_profile(
        input_dir,
        args.output_dir,
        context_profile=profile,
        work_dir=args.output_dir,
        well_names=well_names,
        pattern=args.pattern,
        recursive=args.recursive,
        tops_file=tops_file,
        lithology_mapping_file=lithology_mapping_file,
    )
    successful = sum(1 for item in results if item.success)
    wells = sorted({Path(item.source_file).stem for item in results})
    print(f"Dataset:        {dataset}")
    print(f"Context profile:{profile:>11}")
    print(f"Input LAS:      {input_dir}")
    if tops_file is not None:
        print(f"Tops file:      {tops_file}")
        print(f"Litho map:      {lithology_mapping_file}")
    print(f"Output root:    {args.output_dir}")
    print(f"Processed LAS:  {Path(args.output_dir) / 'processed_las'}  (the context's interpretation)")
    print(f"Wells:          {len(wells)} ({', '.join(wells)})")
    print(f"JSON files:     {len(results)}")
    print(f"Success:        {successful}")
    return 0 if successful == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input_dir
    input_dir, well_names, single_file = _input_scope(
        parser, input_path, args.well_names
    )
    dataset = _detect_dataset(parser, args, input_path, input_dir, well_names)

    if dataset in {"costa", "danish"}:
        return _export_profile(parser, args, dataset, input_dir, well_names)
    from_raw = args.from_raw or (
        single_file and not _looks_like_processed_geolink(input_path)
    )
    if from_raw and args.calculation_profile == "geolink":
        return _export_profile(parser, args, dataset, input_dir, well_names)

    # GeoLink context from the package's interpreted LAS (eval_processed), or
    # the GeoLink inputs run with the current calculations.

    args.well_names = well_names
    args.input_dir = input_dir
    if args.tops_file is None:
        args.tops_file = args.input_dir.parent / "NPD stratigraphic picks north sea.xlsx"
    if args.lithology_mapping_file is None:
        args.lithology_mapping_file = args.input_dir.parent / "Litho_Nomenclature_Kingdom.xlsx"

    raw_input_dir = args.raw_input_dir
    context_input_dir = args.input_dir
    processed_dir = None
    if from_raw:
        processed_dir = Path(args.output_dir) / "first_pass_las"
        batch = process_las_directory_with_workflow(
            input_dir=args.input_dir,
            output_dir=processed_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            config=DynamicWorkflowConfig(calculation_profile=args.calculation_profile),
            well_names=args.well_names,
        )
        print(f"First pass LAS: {processed_dir}  ({args.calculation_profile})")
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
        # interpreted LAS; the library default recomputes the flag.
        workflow_config=DynamicWorkflowConfig(
            calculation_profile=args.calculation_profile, reuse_input_resflag=True
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
        processed_las_dir=Path(args.output_dir) / "processed_las",
    )
    mapping_path = write_reference_curve_name_mapping(args.output_dir)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    processed_wells = sorted({Path(item.source_file).stem for item in results if item.source_file})

    if processed_dir is not None:
        print(f"Raw LAS:        {args.input_dir}")
    print("Dataset:        geolink")
    print(f"Profile:        {args.calculation_profile}")
    print(f"Input LAS:      {context_input_dir}")
    print(f"Tops file:      {args.tops_file}")
    print(f"Output root:    {args.output_dir}")
    print(f"Processed LAS:  {Path(args.output_dir) / 'processed_las'}  (the context's interpretation)")
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
