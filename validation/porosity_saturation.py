"""Validate porosity, saturation, and net-pay behavior against reference wells.

The bundled 30/6-5 well provides a core-porosity comparison. Additional dry
and gas reference wells can be supplied with ``--data-dir`` or the
``PETROPHYS_GEOLINK_DIR`` environment variable. Missing optional wells are
reported explicitly and workflow warnings remain visible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from petrophys._internal.analysis_helpers import depth_step_array, depth_to_meters
from petrophys.io import read_las
from petrophys.workflow.dynamic_workflow import run_dynamic_workflow

BUNDLED_DATA_DIR = SRC_ROOT / "petrophys" / "datasets" / "geolink"

# Expected outcomes are screening labels from the project reference review.
# They are not automatic acceptance criteria: the report preserves the measured
# values so scientific thresholds can be reviewed explicitly.
DEFAULT_WELLS: Mapping[str, str] = {
    "30_6-5": "core",
    "15_9-23": "dry",
    "16_7-6": "dry",
    "25_10-9": "dry",
    "16_10-3": "dry",
    "16_10-5": "dry",
    "25_6-3": "dry",
    "34_11-1": "gas",
    "34_12-1": "gas",
    "35_11-10": "gas",
}
_VALID_EXPECTATIONS = {"core", "dry", "gas"}


def _search_roots(data_dir: Optional[Path] = None) -> Tuple[Path, ...]:
    """Return user-supplied and bundled LAS search roots in priority order."""
    roots = []
    if data_dir is not None:
        roots.append(Path(data_dir).expanduser())
    roots.append(BUNDLED_DATA_DIR)
    return tuple(dict.fromkeys(roots))


def _find(name: str, search_roots: Iterable[Path]) -> Optional[Path]:
    """Find a reference LAS by well name without relying on machine paths."""
    for root in search_roots:
        path = Path(root) / f"{name}.las"
        if path.is_file():
            return path
    return None


def _curve_data(well: Any, *names: str) -> Optional[np.ndarray]:
    for name in names:
        curve = well.get(name)
        if curve is not None:
            return np.asarray(curve.data, dtype=float)
    return None


def _finite_median(values: Optional[np.ndarray]) -> Optional[float]:
    if values is None:
        return None
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _flagged_thickness_m(
    depth: np.ndarray,
    flags: Optional[np.ndarray],
    depth_unit: str,
) -> Optional[float]:
    """Sum flagged sample thickness in meters for either depth direction."""
    if flags is None:
        return None
    depth_m = depth_to_meters(depth, depth_unit)
    flag_values = np.asarray(flags, dtype=float)
    if depth_m.shape != flag_values.shape:
        raise ValueError("Depth and flag arrays must have the same shape")
    thickness = depth_step_array(depth_m)
    selected = np.isfinite(flag_values) & (flag_values > 0.5)
    return float(np.sum(thickness[selected]))


def validate_well(
    name: str,
    expectation: str,
    *,
    search_roots: Sequence[Path],
    lithology: str = "clastic",
) -> Dict[str, Any]:
    """Run the dynamic workflow and return validation metrics for one well."""
    if expectation not in _VALID_EXPECTATIONS:
        raise ValueError(
            f"Unknown expectation {expectation!r}; expected one of "
            f"{sorted(_VALID_EXPECTATIONS)}"
        )

    path = _find(name, search_roots)
    if path is None:
        return {
            "well": name,
            "expectation": expectation,
            "status": "missing",
            "message": "LAS not found in configured search roots",
        }

    well = read_las(path)
    workflow_result = run_dynamic_workflow(well, lithology=lithology)
    depth = np.asarray(well.depth, dtype=float)
    finite_depth = depth_to_meters(depth, well.depth_unit)
    finite_depth = finite_depth[np.isfinite(finite_depth)]

    net_pay = _curve_data(well, "NET_PAY")
    net_reservoir = _curve_data(well, "NET_RES")
    phie = _curve_data(well, "PHIE_CC", "PHIE")
    report: Dict[str, Any] = {
        "well": name,
        "expectation": expectation,
        "status": "validated",
        "source": str(path.resolve()),
        "depth_unit": well.depth_unit,
        "gross_m": (
            float(np.max(finite_depth) - np.min(finite_depth))
            if finite_depth.size
            else None
        ),
        "net_pay_m": _flagged_thickness_m(depth, net_pay, well.depth_unit),
        "net_reservoir_m": _flagged_thickness_m(
            depth,
            net_reservoir,
            well.depth_unit,
        ),
        "phie_p50": _finite_median(phie),
        "workflow": {
            "steps_executed": [step.step for step in workflow_result.steps_executed],
            "steps_skipped": list(workflow_result.steps_skipped),
            "warnings": list(workflow_result.warnings),
        },
    }

    # PHIC is helium/core total porosity, so its direct comparison is PHIT.
    phic = _curve_data(well, "PHIC")
    phit = _curve_data(well, "PHIT")
    if phic is not None and phit is not None:
        core_mask = np.isfinite(phic) & np.isfinite(phit)
        if np.any(core_mask):
            difference = phit[core_mask] - phic[core_mask]
            report["core_comparison"] = {
                "samples": int(np.count_nonzero(core_mask)),
                "phit_p50": float(np.median(phit[core_mask])),
                "phic_p50": float(np.median(phic[core_mask])),
                "bias": float(np.mean(difference)),
                "mae": float(np.mean(np.abs(difference))),
                "phie_p50": (
                    float(np.median(phie[core_mask]))
                    if phie is not None and np.any(np.isfinite(phie[core_mask]))
                    else None
                ),
            }
    return report


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    validated = [row for row in rows if row.get("status") == "validated"]
    missing = [row for row in rows if row.get("status") == "missing"]
    summary: Dict[str, Any] = {
        "requested": len(rows),
        "validated": len(validated),
        "missing": len(missing),
    }
    for expectation in ("dry", "gas"):
        values = [
            float(row["net_pay_m"])
            for row in validated
            if row.get("expectation") == expectation
            and row.get("net_pay_m") is not None
        ]
        if values:
            summary[f"{expectation}_net_pay_median"] = float(np.median(values))
            summary[f"{expectation}_net_pay_min"] = float(np.min(values))
            summary[f"{expectation}_net_pay_max"] = float(np.max(values))
    return summary


def run_validation(
    expectations: Mapping[str, str] = DEFAULT_WELLS,
    *,
    data_dir: Optional[Path] = None,
    lithology: str = "clastic",
) -> Dict[str, Any]:
    """Validate configured reference wells and return a JSON-safe report."""
    search_roots = _search_roots(data_dir)
    rows = [
        validate_well(
            name,
            expectation,
            search_roots=search_roots,
            lithology=lithology,
        )
        for name, expectation in expectations.items()
    ]
    return {
        "configuration": {
            "search_roots": [str(path.resolve()) for path in search_roots],
            "lithology": lithology,
        },
        "wells": rows,
        "summary": _summary(rows),
    }


def _number(value: Any, width: int, precision: int = 1) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{float(value):{width}.{precision}f}"


def format_report(report: Mapping[str, Any]) -> str:
    """Render a compact terminal table from a validation report."""
    lines = [
        "",
        f"{'well':9s} {'exp':5s} {'status':9s} {'net_pay_m':>9s} "
        f"{'net_res_m':>9s} {'gross_m':>8s} {'PHIE50':>7s}  notes",
        "-" * 92,
    ]
    for row in report["wells"]:
        if row["status"] != "validated":
            lines.append(
                f"{row['well']:9s} {row['expectation']:5s} "
                f"{row['status']:9s}  {row.get('message', '')}"
            )
            continue
        note = ""
        core = row.get("core_comparison")
        if core:
            note = (
                f"PHIT={core['phit_p50']:.3f} vs PHIC={core['phic_p50']:.3f}; "
                f"bias={core['bias']:+.3f}, MAE={core['mae']:.3f}"
            )
        lines.append(
            f"{row['well']:9s} {row['expectation']:5s} {row['status']:9s} "
            f"{_number(row.get('net_pay_m'), 9)} "
            f"{_number(row.get('net_reservoir_m'), 9)} "
            f"{_number(row.get('gross_m'), 8, 0)} "
            f"{_number(row.get('phie_p50'), 7, 3)}  {note}"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            f"Validated {summary['validated']} of {summary['requested']} wells; "
            f"{summary['missing']} optional wells missing.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Directory containing additional GeoLink LAS files. Defaults to "
            "PETROPHYS_GEOLINK_DIR when set."
        ),
    )
    parser.add_argument(
        "--well",
        action="append",
        choices=tuple(DEFAULT_WELLS),
        help="Validate only this reference well; repeat for multiple wells.",
    )
    parser.add_argument("--lithology", default="clastic")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_dir = os.environ.get("PETROPHYS_GEOLINK_DIR")
    data_dir = args.data_dir or (Path(env_dir) if env_dir else None)
    if data_dir is not None and not data_dir.is_dir():
        parser.error(f"LAS data directory does not exist: {data_dir}")

    selected = (
        {name: DEFAULT_WELLS[name] for name in args.well}
        if args.well
        else DEFAULT_WELLS
    )
    report = run_validation(
        selected,
        data_dir=data_dir,
        lithology=args.lithology,
    )
    print(format_report(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    return 0 if report["summary"]["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
