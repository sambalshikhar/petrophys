"""
Generate fresh per-well workflow_json trace files without exporting LAS or reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from petrophys.workflow import write_workflow_json_traces_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dynamic workflow and write one detailed workflow_json trace per LAS well. "
            "No LAS files or report files are written."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing LAS files to trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where workflow_json will be written. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--pattern",
        default="*.las",
        help="Glob pattern used to select LAS files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --input-dir.",
    )
    parser.add_argument(
        "--well",
        dest="well_names",
        action="append",
        default=[],
        help="Limit processing to a well stem/name. Repeat for multiple wells.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.input_dir
    results = write_workflow_json_traces_directory(
        input_dir=args.input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        well_names=args.well_names,
    )

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    print(f"Input LAS:      {args.input_dir}")
    print(f"Trace root:     {output_dir / 'workflow_json'}")
    print(f"Pattern:        {args.pattern}")
    print(f"Recursive:      {args.recursive}")
    if args.well_names:
        print(f"Wells filter:   {', '.join(args.well_names)}")
    print(f"Wells traced:   {total}")
    print(f"Success:        {successful}")
    print(f"Failed:         {failed}")
    if failed:
        print("\nFailures:")
        for item in results:
            if item.success:
                continue
            print(f"- {Path(item.source_file).stem}: {' | '.join(item.errors) or 'unknown error'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
