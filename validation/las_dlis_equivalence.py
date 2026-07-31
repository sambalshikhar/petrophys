"""Compare matched LAS and DLIS deliveries through the dynamic workflow."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from petrophys.io import read_dlis_merged, read_las  # noqa: E402
from petrophys.model import Well  # noqa: E402
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow  # noqa: E402

FIXTURE_ROOT = REPO_ROOT / "integration_data" / "daydream"
DEFAULT_LAS = FIXTURE_ROOT / "daydream1_shallow_aps_pex_hngs.las"
DEFAULT_DLIS = FIXTURE_ROOT / "daydream1_shallow_aps_pex_hngs.dlis"
DEEP_LAS = FIXTURE_ROOT / "daydream1_deep_pex_aps_ppc_hngs_ecs.las"
DEEP_DLIS = FIXTURE_ROOT / "daydream1_deep_pex_aps_ppc_ecs.dlis"

RAW_CURVE_PAIRS = {
    "BS": ("BS", "BS"),
    "SGR": ("SGR", "SGR"),
    "GR_EDTC": ("GR:1", "GR_EDTC"),
    "TNPH_2IN": ("TNPH:1", "TNPH"),
    "TNPH_6IN": ("TNPH:2", "TNPH"),
    "APLC": ("APLC", "APLC"),
    "RHO8": ("RHO8", "RHO8"),
    "RHOZ": ("RHOZ", "RHOB"),
    "PEF8": ("PEF8", "PEF"),
    "PEFZ": ("PEFZ", "PEFZ"),
    "RXO8": ("RXO8", "RXO8"),
    "RXOZ": ("RXOZ", "RXO"),
    "CALI": ("CALI", "CALI"),
    "CTEM": ("CTEM", "CTEM"),
    "TENS": ("TENS", "TENS"),
}

MATCHED_WORKFLOW_INPUTS = {
    "GR": ("SGR", "SGR"),
    "NPHI": ("TNPH:2", "TNPH"),
    "RHOB": ("RHOZ", "RHOB"),
    "PEF": ("PEFZ", "PEFZ"),
    "RXO": ("RXOZ", "RXO"),
    "CALI": ("CALI", "CALI"),
    "BS": ("BS", "BS"),
}

DEEP_RAW_CURVE_PAIRS = {
    "BS": ("BS", "BS"),
    "SGR": ("SGR", "SGR"),
    "NPHI": ("NPHI", "NPHI"),
    "TNPH_6IN": ("TNPH:1", "TNPH"),
    "RHO8": ("RHO8:1", "RHOB"),
    "PEF8": ("PEF8:1", "PEF"),
    "RXOZ": ("RXOZ:1", "RXO"),
    "CALI": ("CALI:2", "CALI"),
    "RVDRU_UNCALIBRATED": ("RVDRU:2", "RVDRU"),
    "RVSRU_UNCALIBRATED": ("RVSRU:1", "RVSRU"),
}

DEEP_MATCHED_WORKFLOW_INPUTS = {
    "GR": ("SGR", "SGR"),
    "NPHI": ("NPHI", "NPHI"),
    "RHOB": ("RHO8:1", "RHOB"),
    "PEF": ("PEF8:1", "PEF"),
    "RXO": ("RXOZ:1", "RXO"),
    "CALI": ("CALI:2", "CALI"),
    "BS": ("BS", "BS"),
}

INTERPRETATION_CURVES = (
    "GR",
    "VCL",
    "PHID",
    "PHIN",
    "PHIT",
    "PHIE",
    "SW",
    "SXO",
    "BVW",
    "SWIRR",
    "PERM",
    "VCLAY",
    "VQUAR",
    "VCALC",
    "VDOLO",
    "VANHY",
)


def _finite_depth_range(well: Any, mnemonic: str) -> list[float] | None:
    curve = well.get(mnemonic)
    if curve is None:
        return None
    depth = np.asarray(well.depth, dtype=float)
    values = np.asarray(curve.data, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    if not np.any(valid):
        return None
    return [float(np.min(depth[valid])), float(np.max(depth[valid]))]


def _curve_comparison(
    las_well: Any,
    dlis_well: Any,
    las_name: str,
    dlis_name: str,
) -> dict[str, Any]:
    las_curve = las_well.get(las_name)
    dlis_curve = dlis_well.get(dlis_name)
    if las_curve is None or dlis_curve is None:
        return {
            "status": "missing",
            "las_present": las_curve is not None,
            "dlis_present": dlis_curve is not None,
        }

    las_depth = np.asarray(las_well.depth, dtype=float)
    dlis_depth = np.asarray(dlis_well.depth, dtype=float)
    las_values = np.asarray(las_curve.data, dtype=float)
    dlis_values = np.asarray(dlis_curve.data, dtype=float)
    dlis_valid = np.isfinite(dlis_depth) & np.isfinite(dlis_values)
    if np.count_nonzero(dlis_valid) < 2:
        return {"status": "insufficient_dlis_values"}

    order = np.argsort(dlis_depth[dlis_valid])
    source_depth = dlis_depth[dlis_valid][order]
    source_values = dlis_values[dlis_valid][order]
    source_depth, unique_indices = np.unique(source_depth, return_index=True)
    source_values = source_values[unique_indices]
    inside = (
        np.isfinite(las_depth)
        & np.isfinite(las_values)
        & (las_depth >= source_depth[0])
        & (las_depth <= source_depth[-1])
    )
    if not np.any(inside):
        return {"status": "no_common_finite_depths"}

    reference = las_values[inside]
    candidate = np.interp(las_depth[inside], source_depth, source_values)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[valid]
    candidate = candidate[valid]
    if reference.size == 0:
        return {"status": "no_common_finite_values"}

    difference = candidate - reference
    correlation = None
    if reference.size > 1 and np.std(reference) > 0 and np.std(candidate) > 0:
        correlation = float(np.corrcoef(reference, candidate)[0, 1])
    return {
        "status": "compared",
        "samples": int(reference.size),
        "correlation": correlation,
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "las_mean": float(np.mean(reference)),
        "dlis_mean": float(np.mean(candidate)),
    }


def _step_trace(result: Any) -> dict[str, Any]:
    return {
        step.step: {
            "success": bool(step.success),
            "method": step.method,
            "input_curves": list(step.input_curves),
            "output_curves": list(step.output_curves),
            "message": step.message,
        }
        for step in result.steps_executed
    }


def _matched_input_well(
    source: Any,
    source_names: dict[str, str],
) -> Well:
    """Build a canonical well from explicitly matched physical channels."""
    matched = Well(source.name)
    matched.depth = np.asarray(source.depth, dtype=float).copy()
    matched.depth_unit = source.depth_unit
    matched.header = copy.deepcopy(source.header)
    for canonical, source_name in source_names.items():
        curve = source.get(source_name)
        if curve is None:
            continue
        matched.add_curve(
            canonical,
            np.asarray(curve.data, dtype=float).copy(),
            unit=curve.unit,
            description=f"Matched {source_name}: {curve.metadata.description}",
        )
    return matched


def _run_workflow_pair(
    las_well: Any,
    dlis_well: Any,
    rw: float,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    las_result = run_dynamic_workflow(
        las_well,
        config=DynamicWorkflowConfig(
            rw=rw,
            interpretation_profile="generic_inference",
        ),
    )
    dlis_result = run_dynamic_workflow(
        dlis_well,
        config=DynamicWorkflowConfig(
            rw=rw,
            interpretation_profile="generic_inference",
        ),
    )
    comparisons = {
        name: _curve_comparison(
            las_result.well,
            dlis_result.well,
            name,
            name,
        )
        for name in INTERPRETATION_CURVES
    }
    workflow = {
        "las": {
            "status": las_result.status,
            "profile": las_result.interpretation_profile,
            "steps": _step_trace(las_result),
            "skipped": list(las_result.steps_skipped),
            "warnings": list(las_result.warnings),
            "errors": list(las_result.errors),
        },
        "dlis": {
            "status": dlis_result.status,
            "profile": dlis_result.interpretation_profile,
            "steps": _step_trace(dlis_result),
            "skipped": list(dlis_result.steps_skipped),
            "warnings": list(dlis_result.warnings),
            "errors": list(dlis_result.errors),
        },
    }
    return las_result, dlis_result, workflow, comparisons


def compare_deliveries(
    las_path: Path,
    dlis_path: Path,
    top: float,
    bottom: float,
    rw: float,
    raw_curve_pairs: dict[str, tuple[str, str]] | None = None,
    matched_workflow_inputs: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    raw_curve_pairs = raw_curve_pairs or RAW_CURVE_PAIRS
    matched_workflow_inputs = matched_workflow_inputs or MATCHED_WORKFLOW_INPUTS
    las_full = read_las(las_path)
    dlis_full = read_dlis_merged(dlis_path)
    las_well = las_full.depth_slice(top, bottom)
    dlis_well = dlis_full.depth_slice(top, bottom)
    las_matched = _matched_input_well(
        las_well,
        {canonical: names[0] for canonical, names in matched_workflow_inputs.items()},
    )
    dlis_matched = _matched_input_well(
        dlis_well,
        {canonical: names[1] for canonical, names in matched_workflow_inputs.items()},
    )

    raw_comparisons = {
        label: _curve_comparison(las_well, dlis_well, las_name, dlis_name)
        for label, (las_name, dlis_name) in raw_curve_pairs.items()
    }

    las_result, dlis_result, native_workflow, output_comparisons = _run_workflow_pair(
        las_well, dlis_well, rw
    )
    (
        las_matched_result,
        dlis_matched_result,
        matched_workflow,
        matched_output_comparisons,
    ) = _run_workflow_pair(
        las_matched,
        dlis_matched,
        rw,
    )

    all_output_names = sorted(
        set(las_result.well.curves) | set(dlis_result.well.curves)
    )
    output_coverage = {
        name: {
            "las_finite_depth_range": _finite_depth_range(las_result.well, name),
            "dlis_finite_depth_range": _finite_depth_range(dlis_result.well, name),
        }
        for name in all_output_names
        if name in INTERPRETATION_CURVES
    }
    matched_output_coverage = {
        name: {
            "las_finite_depth_range": _finite_depth_range(
                las_matched_result.well, name
            ),
            "dlis_finite_depth_range": _finite_depth_range(
                dlis_matched_result.well, name
            ),
        }
        for name in INTERPRETATION_CURVES
    }

    dlis_quality = dlis_full.get_parameter("dlis_merge", {}).get("source_quality", {})
    return {
        "inputs": {
            "las": str(las_path),
            "dlis": str(dlis_path),
            "interval": [top, bottom],
            "depth_unit": las_well.depth_unit,
            "rw_ohm_m": rw,
        },
        "reader_summary": {
            "las": {
                "well": las_full.name,
                "samples": las_full.n_samples,
                "curves": las_full.n_curves,
                "depth_range": [las_full.top, las_full.bottom],
            },
            "dlis": {
                "well": dlis_full.name,
                "samples": dlis_full.n_samples,
                "curves": dlis_full.n_curves,
                "depth_range": [dlis_full.top, dlis_full.bottom],
                "lower_quality_channels": {
                    name: details
                    for name, details in dlis_quality.items()
                    if details.get("score", 0) < 0
                },
            },
        },
        "raw_curve_comparison": raw_comparisons,
        "native_delivery": {
            "workflow": native_workflow,
            "interpretation_curve_comparison": output_comparisons,
            "interpretation_curve_coverage": output_coverage,
        },
        "matched_common_channels": {
            "channel_sources": {
                canonical: {"las": names[0], "dlis": names[1]}
                for canonical, names in matched_workflow_inputs.items()
            },
            "workflow": matched_workflow,
            "interpretation_curve_comparison": matched_output_comparisons,
            "interpretation_curve_coverage": matched_output_coverage,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--las", type=Path)
    parser.add_argument("--dlis", type=Path)
    parser.add_argument("--top", type=float)
    parser.add_argument("--bottom", type=float)
    parser.add_argument("--rw", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.preset == "deep":
        las_path = args.las or DEEP_LAS
        dlis_path = args.dlis or DEEP_DLIS
        top = args.top if args.top is not None else 3000.0
        bottom = args.bottom if args.bottom is not None else 3010.0
        raw_curve_pairs = DEEP_RAW_CURVE_PAIRS
        matched_workflow_inputs = DEEP_MATCHED_WORKFLOW_INPUTS
    else:
        las_path = args.las or DEFAULT_LAS
        dlis_path = args.dlis or DEFAULT_DLIS
        top = args.top if args.top is not None else 1200.0
        bottom = args.bottom if args.bottom is not None else 1220.0
        raw_curve_pairs = RAW_CURVE_PAIRS
        matched_workflow_inputs = MATCHED_WORKFLOW_INPUTS
    report = compare_deliveries(
        las_path=las_path,
        dlis_path=dlis_path,
        top=top,
        bottom=bottom,
        rw=args.rw,
        raw_curve_pairs=raw_curve_pairs,
        matched_workflow_inputs=matched_workflow_inputs,
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
