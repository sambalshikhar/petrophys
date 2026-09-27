"""
Reusable CLI for generating category-first context JSON packages by well,
formation, and lithology group.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from petrophys.reporting import (  # noqa: E402
    export_reference_json_work_package_by_category_formation_lithology,
    write_reference_curve_name_mapping,
)
from petrophys._internal.legacy_context import (  # noqa: E402
    set_legacy_benchmark_context,
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def build_parser() -> argparse.ArgumentParser:
    input_default = _environment_path("PETROPHYS_DATA_DIR")
    tops_default = _environment_path("PETROPHYS_TOPS_FILE")
    output_default = _environment_path("PETROPHYS_OUTPUT_DIR")
    lithology_default = _environment_path("PETROPHYS_LITHO_FILE")

    parser = argparse.ArgumentParser(
        description=(
            "Generate category-first JSON context packages from LAS files, formation tops, "
            "and lithology grouping."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=input_default,
        required=input_default is None,
        help="Directory containing source LAS files (or PETROPHYS_DATA_DIR).",
    )
    parser.add_argument(
        "--tops-file",
        type=Path,
        default=tops_default,
        required=tops_default is None,
        help=(
            "CSV or Excel table containing formation tops "
            "(or PETROPHYS_TOPS_FILE)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_default or Path("outputs") / "context_json_by_category",
        help=(
            "Output directory for the generated package. Defaults to "
            "./outputs/context_json_by_category (or PETROPHYS_OUTPUT_DIR)."
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
    parser.add_argument(
        "--lithology-mapping-file",
        type=Path,
        default=lithology_default,
        help=(
            "Optional workbook used to map lithology codes to group labels "
            "(or PETROPHYS_LITHO_FILE). Built-in normalization is used when omitted."
        ),
    )
    parser.add_argument(
        "--reference-structure-dir",
        type=Path,
        default=None,
        help=(
            "Optional root containing <well>_corrected JSON folders. When present, "
            "the exporter reuses those exact formation/lithology intervals and file stems."
        ),
    )
    parser.add_argument(
        "--legacy-benchmark-compat",
        action="store_true",
        help=(
            "Reproduce the archived benchmark context shape: S_V integrated from the "
            "segment top rather than the surface, BS/SO omitted from context curves, "
            "input curves kept out of derived_logs_computed, and the pre-provenance "
            "bad-hole wording. Archived ELECTROFACIES statistics are replayed from the "
            "checksummed v1 manifest because the original k-means state was not recorded."
        ),
    )
    parser.add_argument(
        "--raw-input-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the raw LAS files. When supplied, "
            "input_logs_present and derived/raw classification use these curves. "
            "If omitted and --input-dir ends with _processed, a sibling raw directory is inferred."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    set_legacy_benchmark_context(args.legacy_benchmark_compat)

    results = export_reference_json_work_package_by_category_formation_lithology(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tops_file=args.tops_file,
        pattern=args.pattern,
        recursive=args.recursive,
        minimum_group_thickness=args.min_thickness,
        well_names=args.well_names,
        lithology_mapping_file=args.lithology_mapping_file,
        reference_structure_dir=args.reference_structure_dir,
        raw_input_dir=args.raw_input_dir,
    )
    mapping_path = write_reference_curve_name_mapping(args.output_dir)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    processed_wells = sorted({Path(item.source_file).stem for item in results if item.source_file})

    print(f"Input LAS:      {args.input_dir}")
    print(f"Tops file:      {args.tops_file}")
    print(f"Output root:    {args.output_dir}")
    print(f"Pattern:        {args.pattern}")
    print(f"Recursive:      {args.recursive}")
    print(f"Min thickness:  {args.min_thickness}")
    print(f"Litho map:      {args.lithology_mapping_file}")
    print(f"Reference dir:  {args.reference_structure_dir}")
    print(f"Raw input LAS:  {args.raw_input_dir or 'auto'}")
    print(f"Legacy compat:  {args.legacy_benchmark_compat}")
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
    raise SystemExit(main())
