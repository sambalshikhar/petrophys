import copy
import hashlib
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pytest

from petrophys.datasets import integration_fixture
from petrophys.io import read_dlis_merged, read_las
from petrophys.io.dlis_reader import HAS_DLISIO
from petrophys.model import Well
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

MATCHED_LAS = integration_fixture("daydream1_shallow_aps_pex_hngs.las")
MATCHED_DLIS = integration_fixture("daydream1_shallow_aps_pex_hngs.dlis")
DEEP_MATCHED_LAS = integration_fixture("daydream1_deep_curated_scalar_logs.las")
DEEP_MATCHED_DLIS = integration_fixture("daydream1_deep_pex_aps_ppc_ecs.dlis")

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


def _curve_comparison(
    las_well: Any,
    dlis_well: Any,
    las_name: str,
    dlis_name: str,
) -> dict[str, Any]:
    las_curve = las_well.get(las_name)
    dlis_curve = dlis_well.get(dlis_name)
    if las_curve is None or dlis_curve is None:
        return {"status": "missing"}

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
        "correlation": correlation,
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
    }


def _matched_input_well(source: Any, source_names: dict[str, str]) -> Well:
    matched = Well(source.name)
    matched.depth = np.asarray(source.depth, dtype=float).copy()
    matched.depth_unit = source.depth_unit
    matched.header = copy.deepcopy(source.header)
    for canonical, source_name in source_names.items():
        curve = source.get(source_name)
        if curve is not None:
            matched.add_curve(
                canonical,
                np.asarray(curve.data, dtype=float).copy(),
                unit=curve.unit,
                description=f"Matched physical {canonical} channel",
            )
    return matched


def _workflow_pair(las_well: Well, dlis_well: Well, rw: float):
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

    def summary(result):
        return {
            "steps": [step.step for step in result.steps_executed],
            "skipped": list(result.steps_skipped),
            "errors": list(result.errors),
        }

    comparisons = {
        name: _curve_comparison(las_result.well, dlis_result.well, name, name)
        for name in INTERPRETATION_CURVES
    }
    return {"las": summary(las_result), "dlis": summary(dlis_result)}, comparisons


def _compare_deliveries(
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
    las_well = read_las(las_path).depth_slice(top, bottom)
    dlis_well = read_dlis_merged(dlis_path).depth_slice(top, bottom)
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
    native_workflow, _ = _workflow_pair(las_well, dlis_well, rw)
    matched_workflow, matched_comparisons = _workflow_pair(
        las_matched,
        dlis_matched,
        rw,
    )
    return {
        "raw_curve_comparison": raw_comparisons,
        "native_delivery": {"workflow": native_workflow},
        "matched_common_channels": {
            "workflow": matched_workflow,
            "interpretation_curve_comparison": matched_comparisons,
        },
    }


def _skip_without_real_fixture(path: Path, *, requires_dlisio: bool = True) -> None:
    if requires_dlisio and not HAS_DLISIO:
        pytest.skip("dlisio is not installed")
    if not path.exists():
        pytest.skip(f"Integration fixture is unavailable: {path}; run 'git lfs pull'")
    if path.stat().st_size < 1024:
        pytest.skip(
            f"Integration fixture is a Git LFS pointer: {path}; run 'git lfs pull'"
        )


def test_daydream_deep_curated_las_preserves_the_complete_depth_vector():
    _skip_without_real_fixture(DEEP_MATCHED_LAS, requires_dlisio=False)

    las = lasio.read(DEEP_MATCHED_LAS)
    depth = np.asarray(las.index, dtype="<f8")

    assert len(las.curves) == 121
    assert depth.size == 8548
    assert depth[0] == pytest.approx(2844.8508)
    assert depth[-1] == pytest.approx(4147.4136)
    assert hashlib.sha256(depth.tobytes()).hexdigest() == (
        "0dbeb678b977e1a584ec0b1721b7c5e18940d9f9b2f3bb063c806b3f33a32190"
    )
    assert not any("[" in mnemonic for mnemonic in las.keys())
    assert {
        "BS",
        "SGR",
        "NPHI",
        "RHO8:1",
        "PEF8:1",
        "CALI:2",
        "RVDRU:2",
        "WCLA_WALK2",
    } <= set(las.keys())
    assert las.params["CURATED"].value == "YES"
    assert las.params["CURSHA"].value == (
        "cc63cc99bcdfa12af4580d7993a0f0e530b7395180bc29ec3b55ab1a07914a42"
    )


def test_daydream_shallow_las_and_dlis_workflows_are_format_equivalent():
    _skip_without_real_fixture(MATCHED_LAS)
    _skip_without_real_fixture(MATCHED_DLIS)

    report = _compare_deliveries(
        MATCHED_LAS,
        MATCHED_DLIS,
        top=1200.0,
        bottom=1220.0,
        rw=0.05,
    )

    raw = report["raw_curve_comparison"]
    for name in ["SGR", "TNPH_6IN", "RHOZ", "PEFZ", "RXOZ", "CALI"]:
        assert raw[name]["status"] == "compared"
        assert raw[name]["correlation"] > 0.999
    assert raw["BS"]["mean_absolute_difference"] == 0.0

    native = report["native_delivery"]["workflow"]
    las_steps = set(native["las"]["steps"])
    dlis_steps = set(native["dlis"]["steps"])
    assert native["las"]["errors"] == []
    assert native["dlis"]["errors"] == []
    assert las_steps == dlis_steps
    assert {"vclay", "porosity", "lithology", "permeability"} <= las_steps
    assert "saturation" in native["las"]["skipped"]
    assert "saturation" in native["dlis"]["skipped"]

    matched = report["matched_common_channels"]
    assert set(matched["workflow"]["las"]["steps"]) == set(
        matched["workflow"]["dlis"]["steps"]
    )
    comparisons = matched["interpretation_curve_comparison"]
    for name in [
        "GR",
        "VCL",
        "PHID",
        "PHIN",
        "PHIT",
        "PHIE",
        "PERM",
        "VCLAY",
        "VQUAR",
        "VCALC",
        "VDOLO",
    ]:
        assert comparisons[name]["status"] == "compared"
        assert comparisons[name]["correlation"] > 0.995

    for name in ["VCL", "PHID", "PHIN", "PHIT", "PHIE", "VCLAY"]:
        assert comparisons[name]["mean_absolute_difference"] < 0.001
    for name in ["VQUAR", "VCALC", "VDOLO", "VANHY"]:
        assert comparisons[name]["mean_absolute_difference"] < 0.001
    assert comparisons["PERM"]["mean_absolute_difference"] < 1.0


def test_daydream_deep_las_and_dlis_distinguish_format_from_delivery_content():
    _skip_without_real_fixture(DEEP_MATCHED_LAS)
    _skip_without_real_fixture(DEEP_MATCHED_DLIS)

    report = _compare_deliveries(
        DEEP_MATCHED_LAS,
        DEEP_MATCHED_DLIS,
        top=3000.0,
        bottom=3010.0,
        rw=0.05,
        raw_curve_pairs=DEEP_RAW_CURVE_PAIRS,
        matched_workflow_inputs=DEEP_MATCHED_WORKFLOW_INPUTS,
    )

    raw = report["raw_curve_comparison"]
    for name in [
        "SGR",
        "NPHI",
        "TNPH_6IN",
        "RHO8",
        "PEF8",
        "RXOZ",
        "CALI",
        "RVDRU_UNCALIBRATED",
        "RVSRU_UNCALIBRATED",
    ]:
        assert raw[name]["status"] == "compared"
        assert raw[name]["correlation"] > 0.999999
    assert raw["BS"]["mean_absolute_difference"] == 0.0

    native = report["native_delivery"]["workflow"]
    assert native["las"]["errors"] == []
    assert native["dlis"]["errors"] == []
    assert "saturation" in native["las"]["skipped"]
    assert "saturation" in native["dlis"]["steps"]
    assert "net_pay" in native["las"]["skipped"]
    assert "net_pay" in native["dlis"]["steps"]

    matched = report["matched_common_channels"]
    las_steps = set(matched["workflow"]["las"]["steps"])
    dlis_steps = set(matched["workflow"]["dlis"]["steps"])
    assert las_steps == dlis_steps
    assert "saturation" in matched["workflow"]["las"]["skipped"]
    assert "saturation" in matched["workflow"]["dlis"]["skipped"]

    comparisons = matched["interpretation_curve_comparison"]
    for name in [
        "GR",
        "VCL",
        "PHID",
        "PHIN",
        "PHIT",
        "PHIE",
        "PERM",
        "VCLAY",
        "VQUAR",
        "VCALC",
        "VDOLO",
    ]:
        assert comparisons[name]["status"] == "compared"
        assert comparisons[name]["correlation"] > 0.99999

    for name in ["VCL", "PHID", "PHIN", "PHIT", "PHIE", "VCLAY"]:
        assert comparisons[name]["mean_absolute_difference"] < 0.0001
    for name in ["VQUAR", "VCALC", "VDOLO", "VANHY"]:
        assert comparisons[name]["mean_absolute_difference"] < 0.0001
    assert comparisons["PERM"]["mean_absolute_difference"] < 0.1
