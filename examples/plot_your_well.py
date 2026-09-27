"""Plot a well with optional geology using only public Petrophys APIs."""

import argparse
import petrophys as pp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("well", nargs="?", help="LAS/DLIS/CSV/TSV path; defaults to bundled 30/6-5")
    parser.add_argument("--formations", help="Optional formation table")
    parser.add_argument("--lithology", help="Optional lithology interval table")
    parser.add_argument("--depth-column", default="DEPTH")
    parser.add_argument("--depth-unit", choices=["m", "ft"])
    parser.add_argument("--top", type=float)
    parser.add_argument("--bottom", type=float)
    parser.add_argument("--output", help="Save instead of displaying a window")
    args = parser.parse_args(argv)
    if (args.top is None) != (args.bottom is None):
        parser.error("--top and --bottom must be supplied together")
    path = args.well or pp.sample("30_6-5")
    options = {}
    if str(path).lower().endswith((".csv", ".tsv")):
        options = {"depth_column": args.depth_column}
        if args.depth_unit is not None:
            options["depth_unit"] = args.depth_unit
    well = pp.read(path, formations=args.formations, lithology=args.lithology, **options)
    depth_range = (args.top, args.bottom) if args.top is not None else None
    viewer = pp.plot(well, depth_range=depth_range)
    try:
        if args.output:
            viewer.save(args.output, dpi=180)
        else:
            viewer.show()
        for warning in viewer.warnings:
            print(warning)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
