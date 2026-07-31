from pathlib import Path

import pytest

from petrophys.io.dlis_reader import HAS_DLISIO
from validation.las_dlis_equivalence import (
    DEEP_MATCHED_WORKFLOW_INPUTS,
    DEEP_RAW_CURVE_PAIRS,
    compare_deliveries,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "integration_data" / "daydream"
MATCHED_LAS = FIXTURE_DIR / "daydream1_shallow_aps_pex_hngs.las"
MATCHED_DLIS = FIXTURE_DIR / "daydream1_shallow_aps_pex_hngs.dlis"
DEEP_MATCHED_LAS = FIXTURE_DIR / "daydream1_deep_pex_aps_ppc_hngs_ecs.las"
DEEP_MATCHED_DLIS = FIXTURE_DIR / "daydream1_deep_pex_aps_ppc_ecs.dlis"


def _skip_without_real_fixture(path: Path) -> None:
    if not HAS_DLISIO:
        pytest.skip("dlisio is not installed")
    if not path.exists():
        pytest.skip(f"Integration fixture is unavailable: {path}; run 'git lfs pull'")
    if path.stat().st_size < 1024:
        pytest.skip(
            f"Integration fixture is a Git LFS pointer: {path}; run 'git lfs pull'"
        )


def test_daydream_shallow_las_and_dlis_workflows_are_format_equivalent():
    _skip_without_real_fixture(MATCHED_LAS)
    _skip_without_real_fixture(MATCHED_DLIS)

    report = compare_deliveries(
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

    report = compare_deliveries(
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
