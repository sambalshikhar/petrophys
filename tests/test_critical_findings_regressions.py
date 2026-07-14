import numpy as np
import pandas as pd
import pytest
from defusedxml.common import DefusedXmlException

from petrophys import Well
from petrophys.imputation import PretrainedCurveImputer
from petrophys.imputation.training import (
    CurveFeatureTransformer,
    WELL_ID_COLUMN,
    _standardize_training_values,
    train_pretrained_model_pack,
)
from petrophys.io import read_csv, read_csv_wells
from petrophys.io.dlis_reader import _channel_quality_score
from petrophys.lithology.multi_mineral_solver import MultiMineralSolver
from petrophys.nmr import NMRData
from petrophys.qaqc.standardization import standardize_curves
from petrophys.visualization.log_viewer import load_log_viewer_template
from petrophys.workflow.dynamic_workflow import (
    DynamicWorkflow,
    DynamicWorkflowConfig,
    StepResult,
    WorkflowStep,
)


class _FirstFeatureRegressor:
    def predict(self, frame):
        return np.asarray(frame, dtype=float)[:, 0]


class _FakeChannel:
    def __init__(self, name, long_name="", properties=None):
        self.name = name
        self.long_name = long_name
        self.properties = properties or []


def test_workflow_normalization_converts_units_without_losing_vendor_names():
    well = Well("RAW-UNITS")
    well.depth = np.array([1000.0, 1001.0])
    well.add_curve("RHOB", [2300.0, 2400.0], "kg/m3")
    well.add_curve("NPHI", [20.0, 25.0], "%")
    well.add_curve("GR", [50.0, 60.0], "gAPI")

    result = DynamicWorkflow().run_selected_steps(
        well, [WorkflowStep.NORMALIZATION]
    )

    assert result.success
    assert set(well.curves) == {"RHOB", "NPHI", "GR"}
    assert well["RHOB"].unit == "g/cm3"
    assert well["NPHI"].unit == "v/v"
    np.testing.assert_allclose(well["RHOB"].data, [2.3, 2.4])
    np.testing.assert_allclose(well["NPHI"].data, [0.20, 0.25])


def test_workflow_preserves_a_step_failure_and_does_not_default_rw():
    well = Well("NO-RW")
    well.depth = np.arange(4.0)
    well.add_curve("GR", [40.0, 50.0, 60.0, 70.0], "API")

    workflow = DynamicWorkflow(DynamicWorkflowConfig())
    workflow._step_normalization = lambda _: StepResult(
        step="normalization", success=False, message="deliberate failure"
    )
    selected = workflow.run_selected_steps(well, [WorkflowStep.NORMALIZATION])
    assert selected.success is False

    result = DynamicWorkflow(DynamicWorkflowConfig()).run(well)
    rw_step = next(step for step in result.steps_executed if step.step == "rw")
    assert rw_step.method == "unavailable"
    assert "RW" not in well


def test_workflow_does_not_count_temperature_or_qaqc_as_interpretation():
    well = Well("UNSUPPORTED")
    well.depth = np.arange(4.0)
    well.add_curve("MYSTERY", [1.0, 2.0, 3.0, 4.0], "")

    result = DynamicWorkflow(DynamicWorkflowConfig()).run(well)

    assert result.success
    assert result.status == "no_interpretation"
    assert result.has_usable_interpretation is False


def test_imputer_uses_model_units_and_writes_original_percent_units():
    pack = {
        "schema_version": 2,
        "name": "unit-contract.joblib",
        "unit_contract": "petrophys_standard_units_v1",
        "models": {
            "rf": {
                "PHIE": {
                    "estimator": _FirstFeatureRegressor(),
                    "features": ["NPHI"],
                }
            }
        },
    }
    well = Well("PERCENT")
    well.depth = np.array([1.0, 2.0])
    well.add_curve("PHIE", [10.0, np.nan], "%")
    well.add_curve("NPHI", [20.0, 25.0], "%")

    result = PretrainedCurveImputer(model_pack=pack).impute_missing(
        well, "PHIE", model_name="rf", min_available_features=1
    )

    assert result.imputed_count == 1
    assert well["PHIE"].unit == "%"
    np.testing.assert_allclose(well["PHIE"].data, [10.0, 25.0])


def test_percent_training_units_are_converted_exactly_once():
    values = _standardize_training_values([20.0, 25.0], "NPHI", "%")
    np.testing.assert_allclose(values, [0.20, 0.25])


def test_curve_feature_transformer_follows_sklearn_clone_contract():
    sklearn = pytest.importorskip("sklearn.base")
    features = ("GR", "RDEP")
    cloned = sklearn.clone(CurveFeatureTransformer(features))
    assert cloned.features == features


def test_training_metrics_are_grouped_by_well(monkeypatch, tmp_path):
    rows = 24
    table = pd.DataFrame(
        {
            "GR": np.linspace(30.0, 120.0, rows),
            "NPHI": np.linspace(0.05, 0.35, rows),
            "RHOB": np.linspace(2.1, 2.7, rows),
            WELL_ID_COLUMN: np.repeat(["A", "B", "C"], 8),
        }
    )
    monkeypatch.setattr(
        "petrophys.imputation.training.build_training_table",
        lambda **_: table,
    )

    pack = train_pretrained_model_pack(
        data_dir=tmp_path,
        output_path=tmp_path / "grouped.joblib",
        model_names=["rf"],
        feature_curves=["GR", "NPHI", "RHOB"],
        target_curves=["RHOB"],
        min_samples_per_target=10,
        max_samples_per_target=100,
    )

    metrics = pack["metadata"]["targets"]["RHOB"]["models"]["rf"]
    assert metrics["evaluation"] == "held_out_wells"
    assert metrics["train_wells"] >= 1
    assert metrics["test_wells"] >= 1
    assert pack["training_data"] == tmp_path.name
    assert str(tmp_path.resolve()) not in str(pack)


def test_multi_mineral_solver_closes_without_precomputed_phie():
    result = MultiMineralSolver().solve(
        rhob=np.array([2.55, 2.65]),
        nphi=np.array([0.18, 0.10]),
        vcl=np.array([0.20, 0.05]),
        pef=np.array([3.0, 4.5]),
        dt=np.array([80.0, 65.0]),
    )
    minerals = ["quartz", "calcite", "dolomite", "anhydrite"]
    closure = result["vcl"] + result["phie"]
    closure += sum(result[name] for name in minerals)

    assert np.all(result["solved"])
    np.testing.assert_allclose(closure, 1.0, atol=1.0e-12)


def test_calibrated_dlis_alias_beats_uncalibrated_native_curve():
    calibrated_score, _ = _channel_quality_score(
        _FakeChannel("RVDR", "Calibrated borehole corrected deep resistivity")
    )
    raw_score, _ = _channel_quality_score(
        _FakeChannel("RDEP", "Uncalibrated raw deep resistivity")
    )
    assert calibrated_score > raw_score

    well = Well("DLIS-QUALITY")
    well.depth = np.array([1.0, 2.0])
    well.add_curve("RDEP", [1.0, 1.0], "ohm.m")
    well["RDEP"].metadata.custom["standardization_priority"] = raw_score
    well.add_curve("RVDR", [10.0, 11.0], "ohm.m")
    well["RVDR"].metadata.custom["standardization_priority"] = calibrated_score

    standardize_curves(well)

    np.testing.assert_allclose(well["RDEP"].data, [10.0, 11.0])
    np.testing.assert_allclose(well["RDEP_ALT"].data, [1.0, 1.0])


def test_nmr_missing_bin_rows_remain_missing_and_shapes_are_validated():
    nmr = NMRData(
        depth=np.array([1.0, 2.0]),
        mphi=np.array([np.nan, 6.0]),
        bins=np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]]),
        t2_centers=np.array([4.0, 16.0, 64.0]),
        t2_cutoff=32.0,
    )

    np.testing.assert_allclose(nmr.bvi_from_bins(), [np.nan, 3.0], equal_nan=True)
    np.testing.assert_allclose(nmr.ffi_from_bins(), [np.nan, 3.0], equal_nan=True)
    with pytest.raises(ValueError, match="bins must have shape"):
        NMRData(
            depth=np.array([1.0, 2.0]),
            mphi=np.array([1.0, 2.0]),
            bins=np.ones((2, 2)),
            t2_centers=np.array([4.0, 16.0, 64.0]),
        )


def test_multiwell_csv_requires_explicit_separation(tmp_path):
    path = tmp_path / "multiwell.csv"
    path.write_text(
        "WELL,DEPTH,GR\nA,1,50\nA,2,55\nB,1,70\nB,2,75\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="read_csv_wells"):
        read_csv(path)
    wells = read_csv_wells(path, chunksize=2)
    assert set(wells) == {"A", "B"}
    np.testing.assert_allclose(wells["B"]["GR"].data, [70.0, 75.0])


def test_log_viewer_rejects_xml_entities(tmp_path):
    template = tmp_path / "unsafe.xml"
    template.write_text(
        '<!DOCTYPE template [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<template name="&xxe;"></template>',
        encoding="utf-8",
    )

    with pytest.raises(DefusedXmlException):
        load_log_viewer_template(template_xml_path=template)
