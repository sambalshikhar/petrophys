"""Generate context by interpreting well files and optional geology inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from petrophys.reporting.context import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    export_context,
)


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _add_standard_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dlis-step", type=float,
        help="DLIS merged depth step in the target depth unit (metres by default)",
    )
    parser.add_argument(
        "--lithology-domain",
        choices=("clastic", "carbonate", "evaporite", "auto"),
        help="Broad mineral-solving domain; defaults to LAS metadata or clastic",
    )
    parser.add_argument(
        "--formations",
        type=Path,
        default=_path_from_env("PETROPHYS_FORMATIONS_FILE"),
        help="Optional formation tops or intervals table",
    )
    parser.add_argument(
        "--lithology",
        type=Path,
        default=_path_from_env("PETROPHYS_LITHOLOGY_FILE"),
        help="Optional per-depth lithology intervals table",
    )
    parser.add_argument(
        "--lithology-codes",
        type=Path,
        default=_path_from_env("PETROPHYS_LITHOLOGY_CODES_FILE"),
        help="Optional lithology code-to-label table",
    )


def build_package_parser(layout: str) -> argparse.ArgumentParser:
    """Build the parser shared by the package script entry points."""
    parser = argparse.ArgumentParser(
        description="Compute formation/lithology context packages from LAS or DLIS files"
    )
    input_dir = _path_from_env("PETROPHYS_DATA_DIR")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=input_dir,
        required=input_dir is None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PETROPHYS_OUTPUT_DIR", "outputs/context_package")),
    )
    _add_standard_inputs(parser)
    parser.add_argument("--well", dest="well_names", action="append", default=[])
    parser.add_argument("--pattern", default="*.las", help="Input glob; use *.dlis for DLIS")
    parser.add_argument("--depth-unit", choices=("m", "ft"), help="Required for tabular well logs")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=2.0,
        help="Minimum lithology-group thickness in the well depth unit",
    )
    if layout == "category_formation_lithology":
        parser.add_argument(
            "--profile",
            dest="context_profile",
            choices=(
                "geolink", "geolink_v1", "geolink_train", "geolink_train_v1", "costa", "costa_v1",
                "danish", "danish_v1",
            ),
            help=(
                "Compute a published context package with its own pipeline from the input "
                "directory's raw LAS (GeoLink eval, GeoLink training, COSTA or Danish); geolink, "
                "geolink_train and danish also need --formations (their tops workbook) and "
                "--lithology-codes (the GeoLink codebook). Omit it to interpret with your "
                "supplied settings."
            ),
        )
    parser.add_argument("--tops-rank", choices=("group", "formation", "member"))
    parser.add_argument("--tops-sheet", help="Explicit formation Excel sheet name")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    return parser


def _package_options(args: argparse.Namespace) -> dict:
    options = {
        "tops_file": args.formations,
        "lithology_file": args.lithology,
        "lithology_mapping_file": args.lithology_codes,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "minimum_group_thickness": args.min_thickness,
        "well_names": args.well_names,
        "tops_options": {
            key: value
            for key, value in (
                ("rank", args.tops_rank),
                ("sheet_name", args.tops_sheet),
            )
            if value is not None
        },
    }
    if args.lithology_domain is not None:
        options["lithology_domain"] = args.lithology_domain
    reader_options = _reader_options(args)
    if reader_options:
        options["reader_options"] = reader_options
    return options


def _reader_options(args: argparse.Namespace) -> dict:
    options = {}
    if args.depth_unit:
        options["depth_unit"] = args.depth_unit
    if args.dlis_step is not None:
        if not 0 < args.dlis_step < float("inf"):
            raise ValueError("--dlis-step must be finite and positive")
        options["step"] = args.dlis_step
    return options


def package_main(layout: str, argv=None) -> int:
    parser = build_package_parser(layout)
    args = parser.parse_args(argv)
    try:
        results = export_context(
            args.input_dir,
            args.output_dir,
            layout=layout,
            max_bytes=args.max_bytes,
            max_total_bytes=args.max_total_bytes,
            context_profile=getattr(args, "context_profile", None),
            **_package_options(args),
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"petrophys-context: {exc}\n")
    print(f"Exported {len(results)} contexts to {args.output_dir}")
    return 0


def _single_well_result(args: argparse.Namespace):
    from petrophys.api import interpret, read
    from petrophys.workflow.zoned_batch import lithology_domain_from_metadata

    reader_options = _reader_options(args)
    well = read(
        args.input,
        formations=args.formations,
        lithology=args.lithology,
        **reader_options,
    )
    settings = {}
    metadata_domain = lithology_domain_from_metadata(well)
    if metadata_domain is not None:
        settings.setdefault("lithology_domain", metadata_domain)
    if args.lithology_domain is not None:
        settings["lithology_domain"] = args.lithology_domain
    if args.lithology is not None:
        settings.update(
            lithology_intervals=args.lithology,
            lithology_source="external",
        )
    return interpret(well, **settings).raise_for_errors()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Well file (LAS, DLIS, CSV, TSV) or directory")
    parser.add_argument("output", type=Path, help="Output JSON file or new package directory")
    parser.add_argument(
        "--layout",
        choices=("well", "formation_lithology", "category_formation_lithology"),
        default="well",
    )
    _add_standard_inputs(parser)
    parser.add_argument("--depth-unit", choices=("m", "ft"), help="Required for tabular well logs")
    parser.add_argument("--well", action="append", help="Package well filter; repeat for multiple wells")
    parser.add_argument("--pattern", default="*.las", help="Input glob; use *.dlis for DLIS")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-thickness", type=float, default=2.0)
    parser.add_argument(
        "--profile",
        dest="context_profile",
        choices=(
            "geolink", "geolink_v1", "geolink_train", "geolink_train_v1", "costa", "costa_v1",
            "danish", "danish_v1",
        ),
        help=(
            "Compute a published category context package (geolink, geolink_train, costa or "
            "danish) from its raw LAS directory; geolink, geolink_train and danish also "
            "need --formations and --lithology-codes. Omit it to interpret with your "
            "supplied settings."
        ),
    )
    parser.add_argument("--tops-rank", choices=("group", "formation", "member"))
    parser.add_argument("--tops-sheet", help="Explicit formation Excel sheet name")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a single JSON file; package directories are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        if args.layout == "well":
            if args.context_profile is not None:
                parser.error("--profile requires --layout category_formation_lithology")
            if args.input.is_dir():
                parser.error("layout=well requires one well file")
            if args.well or args.tops_rank or args.tops_sheet:
                parser.error("--well and tops selection apply to package layouts")
            if args.lithology_codes is not None:
                parser.error("--lithology-codes applies only to package layouts")
            result = _single_well_result(args)
            exported = export_context(
                result,
                args.output,
                max_bytes=args.max_bytes,
                overwrite=args.overwrite,
            )
        else:
            package_args = argparse.Namespace(
                **vars(args),
                input_dir=args.input,
                output_dir=args.output,
                well_names=args.well or [],
            )
            exported = export_context(
                args.input,
                args.output,
                layout=args.layout,
                max_bytes=args.max_bytes,
                max_total_bytes=args.max_total_bytes,
                context_profile=args.context_profile,
                **_package_options(package_args),
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"petrophys-context: {exc}\n")
    print(
        exported
        if isinstance(exported, Path)
        else f"Exported {len(exported)} contexts to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
