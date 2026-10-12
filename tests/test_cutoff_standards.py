"""Curriculum-standard net-pay cutoffs: single source of truth and resolution."""

import numpy as np
import pytest

from petrophys.model.constants import DefaultParameters
from petrophys.evaluation import STANDARD_CUTOFFS, standard_cutoffs
from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
from petrophys.workflow import DynamicWorkflowConfig


def test_standard_cutoffs_values():
    assert STANDARD_CUTOFFS["clastic"] == {
        "vcl_cutoff": 0.40, "phie_cutoff": 0.08, "sw_cutoff": 0.60,
    }
    assert STANDARD_CUTOFFS["carbonate"] == {
        "vcl_cutoff": 0.30, "phie_cutoff": 0.045, "sw_cutoff": 0.70,
    }
    # helper normalizes lithology spellings and returns a copy
    got = standard_cutoffs("CARBONATE")
    assert got == STANDARD_CUTOFFS["carbonate"]
    got["sw_cutoff"] = 0.99
    assert STANDARD_CUTOFFS["carbonate"]["sw_cutoff"] == 0.70


def test_netpay_module_resolves_per_lithology():
    clastic = NetPayModule()
    assert (clastic.vcl_cutoff, clastic.phie_cutoff, clastic.sw_cutoff) == (0.40, 0.08, 0.60)
    carb = NetPayModule(lithology="carbonate")
    assert (carb.vcl_cutoff, carb.phie_cutoff, carb.sw_cutoff) == (0.30, 0.045, 0.70)
    # explicit values always win
    explicit = NetPayModule(vcl_cutoff=0.5, phie_cutoff=0.12, sw_cutoff=0.65,
                            lithology="carbonate")
    assert (explicit.vcl_cutoff, explicit.phie_cutoff, explicit.sw_cutoff) == (0.5, 0.12, 0.65)


def test_netpay_module_per_depth_lithology_cutoffs():
    lith = np.array(["clastic", "carbonate", "carbonate", "clastic"])
    mod = NetPayModule(lithology=lith)
    vcl_cut = mod._build_vcl_cutoff(4)
    phie_cut = mod._build_phie_cutoff(4)
    sw_cut = mod._build_sw_cutoff(4)
    np.testing.assert_allclose(vcl_cut, [0.40, 0.30, 0.30, 0.40])
    np.testing.assert_allclose(phie_cut, [0.08, 0.045, 0.045, 0.08])
    np.testing.assert_allclose(sw_cut, [0.60, 0.70, 0.70, 0.60])


def test_dynamic_config_cutoff_defaults():
    from petrophys.workflow.dynamic_workflow import _profile_setting

    # Unset by default: "current" resolves them per lithology family, the
    # benchmark profile to the eval package's VCL 0.3 and SW 0.7.
    cfg = DynamicWorkflowConfig()
    assert (cfg.vcl_cutoff, cfg.sw_cutoff, cfg.phie_cutoff) == (None, None, None)
    benchmark = DynamicWorkflowConfig(calculation_profile="benchmark")
    resolved = tuple(_profile_setting(benchmark, key) for key in ("vcl_cutoff", "sw_cutoff", "phie_cutoff"))
    assert resolved == (0.3, 0.7, None)


def test_constants_match_standards():
    assert DefaultParameters.VCL_CUTOFF_CLASTIC == 0.40
    assert DefaultParameters.VCL_CUTOFF_CARBONATE == 0.30
    assert DefaultParameters.PHIE_CUTOFF_CARBONATE == 0.045
    assert DefaultParameters.SW_CUTOFF_CLASTIC == 0.60
    assert DefaultParameters.SW_CUTOFF_CARBONATE == 0.70


def test_category_exporter_threads_workflow_config(monkeypatch, tmp_path):
    """The category context exporter forwards workflow_config to the zoned runner."""
    import petrophys.workflow.zoned_batch as zb
    from petrophys.reporting import (
        export_reference_json_work_package_by_category_formation_lithology as export_fn,
    )
    from petrophys.workflow import DynamicWorkflowConfig

    captured = {}

    def fake_runner(well, config=None):
        captured["config"] = config
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(zb, "run_dynamic_workflow_with_zones", fake_runner)

    las_dir = tmp_path / "las"
    las_dir.mkdir()
    import petrophys, shutil
    shutil.copy2(petrophys.sample("16_7-1"), las_dir / "16_7-1.las")
    tops = tmp_path / "tops.xlsx"
    import pandas as pd
    pd.DataFrame({"well": ["16_7-1"], "formation": ["A"], "md_top": [1000.0]}).to_excel(
        tops, index=False)

    cfg = DynamicWorkflowConfig(rw=0.123, rw_meas_temp_f=200.0)
    results = export_fn(
        input_dir=las_dir, output_dir=tmp_path / "out", tops_file=tops,
        workflow_config=cfg,
    )
    assert captured["config"] is cfg
    # the capture-raise surfaces as a failed result, not a crash
    assert results and not results[0].success


def test_pretrained_imputer_pack_loads_without_legacy_namespaces():
    """The bundled pack exposes only validated targets and canonical classes."""
    sklearn = pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    pytest.importorskip("xgboost")
    if tuple(int(part) for part in sklearn.__version__.split(".")[:2]) != (1, 8):
        pytest.skip("bundled model requires scikit-learn 1.8.x")

    from petrophys.preprocessing.imputation import PretrainedCurveImputer
    from petrophys.preprocessing.imputation.curve_imputer import DEFAULT_TARGET_CURVES
    from petrophys.model.well import Well

    imputer = PretrainedCurveImputer.from_pretrained()
    assert imputer.target_curves == DEFAULT_TARGET_CURVES == ["DTC", "NPHI", "RHOB"]
    assert {"BS", "CALI"} <= set(imputer.model_pack["feature_curves"])
    for name, models in imputer.model_pack["models"].items():
        assert sorted(models) == imputer.target_curves
        for target, model in models.items():
            transformer = model["estimator"].named_steps["curve_transform"]
            assert type(transformer).__module__.startswith("petrophys.preprocessing")
            metrics = imputer.model_pack["metadata"]["targets"][target]["models"][name]
            assert metrics["evaluation"] == "held_out_wells"
            assert metrics["r2"] >= 0.5

    well = Well("TARGET-SCOPE")
    well.depth = np.array([1000.0, 1001.0])
    for target in ("BS", "CALI"):
        well.add_curve(target, np.array([8.5, np.nan]), "in")
        for name in imputer.model_names:
            with pytest.raises(KeyError, match="not available"):
                imputer.impute_missing(well, target, model_name=name)
        with pytest.raises(ValueError, match="No grouped held-out model"):
            imputer.impute_missing(well, target)
        np.testing.assert_array_equal(well[target].data, [8.5, np.nan])
