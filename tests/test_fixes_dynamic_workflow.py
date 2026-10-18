"""Regression tests for fixes in petrophys.workflow.dynamic_workflow."""

import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import DynamicWorkflow, WorkflowStep


def _well(name="30_6-5", **kwargs):
    return pp.read(pp.sample(name), **kwargs)


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_run_dynamic_workflow_overrides_lithology_domain_on_config():
    config = DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230)
    result = pp.run_dynamic_workflow(_well(), config=config, lithology_domain="carbonate")
    assert result.config.lithology_domain == "carbonate"
    assert config.lithology_domain == "clastic"

    back = pp.run_dynamic_workflow(
        _well(), config=DynamicWorkflowConfig(lithology_domain="carbonate", rw=0.07),
        lithology_domain="clastic",
    )
    assert back.config.lithology_domain == "clastic"


def test_run_selected_steps_leaves_caller_config_unchanged():
    config = DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230)
    well = _well(formations=pp.sample("formations_30_6-5"))
    DynamicWorkflow(config).run_selected_steps(well, [WorkflowStep.VCLAY])
    assert config.formation_tops is None


def test_formation_tops_from_read_formations_table():
    tops = pp.read_formations(pp.sample("formations_30_6-5"))
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, formation_tops=tops)
    assert not [e for e in result.errors if "intervals" in e]
    result.summary()


def test_formation_tops_of_wrong_type_gives_clear_error():
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, formation_tops=123)
    assert any("formation_tops must be FormationTops" in e for e in result.errors)


def test_input_temp_curve_is_kept_and_reported():
    well = _well()
    measured = np.linspace(10.0, 120.0, well.n_samples)
    well.add_curve("TEMP", measured, "degC", "Measured")
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    kept = result.well["TEMP_LOG"]
    assert np.isfinite(kept.data).all()
    # The measured log is the temperature source; TEMP_LOG keeps the raw copy.
    assert np.allclose(kept.data, result.well["TEMP"].data)
    assert any("TEMP_LOG" in w for w in result.warnings)


def test_input_temp_curve_untouched_under_geolink_profile():
    well = _well()
    well.add_curve("TEMP", np.linspace(10.0, 120.0, well.n_samples), "degC", "Measured")
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, calculation_profile="geolink")
    assert result.well.get("TEMP_LOG") is None


def test_explicit_rm_wins_over_header_rmf():
    result = pp.run_dynamic_workflow(
        _well(), DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230, rm=0.2, rm_meas_temp_f=60)
    )
    # An explicit rm wins over the header RMF, as rw does over header RW.
    step = _step(result, "rmf")
    assert step.method.startswith("rm_to_rmf")
    assert any("config.rm" in d for d in step.decisions)


def test_rw_from_rxo_rt_is_warned():
    well = _well()
    well._parameters.pop("RW", None)
    result = pp.interpret(well)
    if _step(result, "rw").method != "rxo_rt_clean_water":
        pytest.skip("Rw did not take the Rxo/Rt route")
    assert any("estimated from Rxo/Rt" in w for w in result.warnings)


def test_porosity_trace_does_not_promise_refinement():
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230)
    decisions = " ".join(_step(result, "porosity").decisions)
    assert "will be refined" not in decisions


def test_removed_input_core_curves_are_warned():
    well = _well("HW-3")
    assert well.get("PERM_CC") is not None
    result = pp.interpret(well, lithology_domain="carbonate")
    assert any("Input curves" in w and "PERM_CC" in w for w in result.warnings)
