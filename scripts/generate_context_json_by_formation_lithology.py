"""Compatibility entry point for formation_lithology context packages."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petrophys.reporting.context_cli import build_package_parser, package_main  # noqa: E402


def build_parser():
    return build_package_parser("formation_lithology")


def main(argv=None):
    return package_main("formation_lithology", argv)


if __name__ == "__main__":
    raise SystemExit(main())
