import hashlib
from pathlib import Path

import lasio
import numpy as np
import pytest

from petrophys.datasets import integration_fixture
from petrophys.io.dlis_reader import HAS_DLISIO
from validation.las_dlis_equivalence import (
    DEEP_MATCHED_WORKFLOW_INPUTS,
    DEEP_RAW_CURVE_PAIRS,
    compare_deliveries,
)

MATCHED_LAS = integration_fixture("daydream1_shallow_aps_pex_hngs.las")
MATCHED_DLIS = integration_fixture("daydream1_shallow_aps_pex_hngs.dlis")
DEEP_MATCHED_LAS = integration_fixture("daydream1_deep_curated_scalar_logs.las")
DEEP_MATCHED_DLIS = integration_fixture("daydream1_deep_pex_aps_ppc_ecs.dlis")


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
