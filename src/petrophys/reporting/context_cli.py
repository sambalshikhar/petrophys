"""Command-line context generation without hidden local input files."""

import argparse
import os
from pathlib import Path

from petrophys.reporting.context import DEFAULT_MAX_BYTES, DEFAULT_MAX_TOTAL_BYTES, export_context


def build_package_parser(layout):
    """Retain the legacy script flags, without nonexistent default workbooks."""
    parser = argparse.ArgumentParser(description="Export bounded formation/lithology context packages")
    for flag, variable in (("input-dir", "PETROPHYS_DATA_DIR"), ("tops-file", "PETROPHYS_TOPS_FILE")):
        value = os.environ.get(variable)
        parser.add_argument("--" + flag, type=Path, default=Path(value) if value else None,
                            required=not bool(value), help=f"Required input path (or {variable})")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("PETROPHYS_OUTPUT_DIR", "outputs/context_package")))
    parser.add_argument("--well", dest="well_names", action="append", default=[])
    parser.add_argument("--pattern", default="*.las")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-thickness", type=float, default=2.0)
    mapping = os.environ.get("PETROPHYS_LITHO_FILE")
    parser.add_argument("--lithology-mapping-file", type=Path, default=Path(mapping) if mapping else None)
    parser.add_argument("--tops-rank", choices=("group", "formation", "member"))
    parser.add_argument("--tops-sheet", help="Explicit Excel sheet name")
    parser.add_argument("--schema-version", choices=("1.0", "legacy"), default="1.0")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--legacy-benchmark-compat", action="store_true", help="Replay archived benchmark settings and legacy JSON layout")
    if layout == "category_formation_lithology":
        parser.add_argument("--reference-structure-dir", type=Path)
        parser.add_argument("--raw-input-dir", type=Path)
        parser.add_argument("--use-finalized-processed-curves", action="store_true", help="Treat input LAS files as finalized interpretation products; do not rerun the workflow")
    return parser


def package_main(layout, argv=None):
    from petrophys._internal.legacy_context import ENV_VAR, set_legacy_benchmark_context
    parser = build_package_parser(layout)
    args = parser.parse_args(argv)
    options = dict(tops_file=args.tops_file, pattern=args.pattern, recursive=args.recursive,
                   minimum_group_thickness=args.min_thickness, well_names=args.well_names,
                   lithology_mapping_file=args.lithology_mapping_file,
                   tops_options={key: value for key, value in (("rank", args.tops_rank), ("sheet_name", args.tops_sheet)) if value is not None})
    if layout == "category_formation_lithology":
        options.update(reference_structure_dir=args.reference_structure_dir, raw_input_dir=args.raw_input_dir,
                       use_finalized_processed_curves=args.use_finalized_processed_curves)
    previous = os.environ.get(ENV_VAR)
    try:
        set_legacy_benchmark_context(args.legacy_benchmark_compat)
        results = export_context(args.input_dir, args.output_dir, layout=layout,
                                 schema_version="legacy" if args.legacy_benchmark_compat else args.schema_version,
                                 max_bytes=args.max_bytes, max_total_bytes=args.max_total_bytes, **options)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"petrophys-context: {exc}\n")
    finally:
        if previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = previous
    print(f"Exported {len(results)} contexts to {args.output_dir}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Well file, or LAS directory for package layouts")
    parser.add_argument("output", type=Path, help="Output JSON file or new package directory")
    parser.add_argument("--layout", choices=("well", "formation_lithology", "category_formation_lithology"), default="well")
    parser.add_argument("--schema-version", choices=("1.0", "legacy"), default="1.0")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--overwrite", action="store_true", help="Replace a single JSON file, never a package directory")
    parser.add_argument("--formations", type=Path, help="Optional formations; required for package layouts")
    parser.add_argument("--lithology", type=Path, help="Optional intervals for a single well")
    parser.add_argument("--depth-unit", choices=("m", "ft"), help="Required for tabular well logs")
    parser.add_argument("--interpret", action="store_true", help="Interpret the well before exporting")
    parser.add_argument("--rw", type=float, help="Measured formation-water resistivity, ohm.m; requires --interpret")
    parser.add_argument("--rw-temp-f", type=float, help="Reference temperature of --rw, degF")
    parser.add_argument("--well", action="append", help="Package well filter; repeat for multiple wells")
    parser.add_argument("--tops-rank", choices=("group", "formation", "member"))
    args = parser.parse_args(argv)
    try:
        options = {}
        data = args.input
        if args.layout == "well":
            from petrophys.api import interpret, read
            if args.well or args.tops_rank:
                parser.error("--well and --tops-rank apply to package exports")
            if not args.interpret and (args.rw is not None or args.rw_temp_f is not None):
                parser.error("--rw and --rw-temp-f require --interpret")
            read_options = {"depth_unit": args.depth_unit} if args.depth_unit else {}
            data = read(data, formations=args.formations, lithology=args.lithology, **read_options)
            original = set(data.curves)
            if args.interpret:
                settings = {}
                if args.rw is not None:
                    settings["rw"] = args.rw
                if args.rw_temp_f is not None:
                    settings["rw_meas_temp_f"] = args.rw_temp_f
                data = interpret(data, **settings).raise_for_errors()
            options.update(source_file=args.input, original_curves=original)
        else:
            if args.interpret or args.rw is not None or args.rw_temp_f is not None or args.lithology or args.depth_unit:
                parser.error("Package calculations use workflow_config in the Python API; single-well flags are not accepted")
            options.update(tops_file=args.formations, well_names=args.well)
            if args.tops_rank:
                options["tops_options"] = {"rank": args.tops_rank}
        result = export_context(data, args.output, layout=args.layout, schema_version=args.schema_version,
                                max_bytes=args.max_bytes, max_total_bytes=args.max_total_bytes,
                                overwrite=args.overwrite, **options)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"petrophys-context: {exc}\n")
    print(result if isinstance(result, Path) else f"Exported {len(result)} contexts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
