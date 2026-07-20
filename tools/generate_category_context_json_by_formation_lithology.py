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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate category-first JSON context packages from LAS files, formation tops, "
            "and lithology grouping."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[1] / "src" / "petrophys" / "datasets" / "geolink")),
        help="Directory containing source LAS files.",
    )
    parser.add_argument(
        "--tops-file",
        type=Path,
        default=Path(os.environ.get("PETROPHYS_TOPS_FILE", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "NPD_stratigraphic_picks.xlsx")),
        help="Excel workbook containing formation tops.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PETROPHYS_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs" / "context_json_by_category")),
        help="Output directory for the generated package.",
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
        default=Path(os.environ.get("PETROPHYS_LITHO_FILE", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "Litho_Nomenclature_Kingdom.xlsx")),
        help="Workbook used to map LITHOLOGY_GEOLINK codes to lithology-group labels.",
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
