"""Regression tests for the workflow fixes (custom, standard, quick look, zoned batch, api)."""

import json
import shutil
import warnings

import lasio
import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys.model.well import Well
from petrophys.workflow import steps
from petrophys.workflow import zoned_batch as zb
from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
from petrophys.workflow.quick_look import QuickLook, quick_look_analysis
from petrophys.workflow.standard_workflow import (
    StandardWorkflow,
    WorkflowConfig,
    run_standard_workflow,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --- custom workflows -------------------------------------------------------

def test_summary_table_net_pay_unknown_without_flag():
    wf = pp.Workflow("vcl", [steps.clay_volume(method="gr")])
    row = wf.run_many([pp.sample("HW-3")]).summary_table().iloc[0]
    assert row["Gross"] > 0
    assert np.isnan(row["Net Pay"]) and np.isnan(row["NTG"])
    outside = wf.run_many([pp.sample("HW-3")]).summary_table(top=0, bottom=10).iloc[0]
    assert outside["Gross"] == 0 and np.isnan(outside["NTG"])


def test_to_las_replaces_previous_provenance(tmp_path):
    two = pp.Workflow("two", [steps.clay_volume(method="gr"),
                              steps.clay_volume(method="gr", name="vcl2")])
    first = two.run(pp.read(pp.sample("HW-3"))).to_las(tmp_path / "a.las")
    one = pp.Workflow("one", [steps.clay_volume(method="gr")])
    second = one.run(pp.read(first)).to_las(tmp_path / "b.las")
    names = [item.mnemonic for item in lasio.read(second).params if item.mnemonic.startswith("PP_")]
    assert names == ["PP_VERSION", "PP_WORKFLOW", "PP_STATUS", "PP_STEP01"]
    assert lasio.read(second).params["PP_WORKFLOW"].value == "one"


def test_failed_custom_step_writes_no_partial_output():
    @pp.workflow_step(requires=["GR"], produces={"A_FIX": "v/v", "B_FIX": "v/v"})
    def half(well):
        return {"A_FIX": np.ones(well.n_samples), "B_FIX": np.ones(3)}

    @pp.workflow_step(requires=["A_FIX"], produces={"C_FIX": "v/v"})
    def uses_a(well):
        return {"C_FIX": well.get("A_FIX").data * 2}

    run = pp.Workflow("h", [half(), uses_a()]).run(pp.read(pp.sample("HW-3")), strict=False)
    assert run.well.get("A_FIX") is None
    assert "C_FIX" not in run.output_curves
    assert "uses_a" in run.steps_skipped


def test_huge_integer_parameter_is_a_validation_error():
    with pytest.raises(ValueError, match="sw_cutoff must be finite"):
        pp.Workflow.from_dict({"steps": [{"kind": "net_pay", "params": {"sw_cutoff": int("9" * 400)}}]})


# --- deprecated StandardWorkflow -----------------------------------------------

def test_standard_workflow_corrects_header_rw_to_formation_temperature():
    well = pp.read(pp.sample("HW-3"))
    result = run_standard_workflow(well, bht_f=251.0, td_ft=8299.0)
    sat = next(step for step in result.steps if step.step == "saturation")
    assert sat.parameters["rw_reference_temp_f"] == pytest.approx(68.0)
    # Rw 0.0682 ohm.m at 68 F is about 0.02 ohm.m at 240-250 F: Sw well below the uncorrected value.
    assert np.nanmedian(well.get("SW").data) < 0.13
    assert any("PHIE_CC" in message and "kept as" in message for message in result.warnings)


def test_standard_workflow_sonic_endpoints_are_used():
    well = pp.read(pp.sample("1046123714"))
    run_standard_workflow(well, calculate_sw=False, calculate_perm=False,
                          calculate_net_pay=False, matrix_dt=47.6, fluid_dt=200.0)
    dt = (well.get("DTC") or well.get("DT")).data
    phis = well.get("PHIS").data
    expected = np.clip((dt - 47.6) / (200.0 - 47.6), 0, 0.5)
    finite = np.isfinite(dt)
    np.testing.assert_allclose(phis[finite], expected[finite])


def test_standard_workflow_success_false_when_steps_fail():
    result = StandardWorkflow().run(Well(name="empty"))
    assert result.success is False


def test_standard_workflow_unknown_kwarg_with_config_raises_and_config_unchanged():
    config = WorkflowConfig()
    with pytest.raises(TypeError):
        run_standard_workflow(pp.read(pp.sample("HW-3")), config=config, rw_typo=0.03)
    run_standard_workflow(pp.read(pp.sample("HW-3")), config=config, calculate_perm=False,
                          calculate_sw=False, calculate_net_pay=False, gr_min=30.0)
    assert config.gr_min == 20.0


def test_standard_workflow_net_pay_trace_names_porosity_used():
    result = run_standard_workflow(pp.read(pp.sample("25_5-1")))
    net = next(step for step in result.steps if step.step == "net_pay")
    assert "PHIE_CC" in net.input_curves
    assert any("PHIE_CC >" in decision for decision in net.decisions)


# --- quick look ----------------------------------------------------------------

def test_quick_look_finds_aliased_resistivity():
    result = quick_look_analysis(pp.read(pp.sample("1046123714")))
    assert result.has_resistivity
    assert np.isfinite(result.estimated_sw)


def test_quick_look_rw_is_configurable():
    well = pp.read(pp.sample("25_5-1"))
    default = quick_look_analysis(well).estimated_sw
    lower = quick_look_analysis(well, rw=0.02).estimated_sw
    assert lower == pytest.approx(default * np.sqrt(0.02 / 0.05))


def test_quick_look_undefined_inputs_are_nan():
    assert np.isnan(QuickLook(gr_clean=50, gr_shale=50)._quick_vcl(60.0))
    assert np.isnan(QuickLook()._quick_sw(0.0, 0.2))


# --- zoned batch -----------------------------------------------------------------

def test_header_salinity_units():
    from petrophys.interpretation.electrical.rw.salinity_rw import salinity_mg_per_l_to_ppm

    assert zb._salinity_to_ppm(150000, "mg/l") == pytest.approx(salinity_mg_per_l_to_ppm(150000))
    assert zb._salinity_to_ppm(150, "g/l") == pytest.approx(salinity_mg_per_l_to_ppm(150000))
    assert zb._salinity_to_ppm(30, "g/kg") == pytest.approx(30000)
    # The "geolink" profile keeps the benchmark reading.
    assert zb._salinity_to_ppm(150000, "mg/l", benchmark=True) == 150000
    overrides = {}
    zb._apply_parameter_alias("SAL", 150000, "mg/l", overrides, {}, [],
                              calculation_profile="geolink")
    assert overrides["salinity_ppm"] == 150000


def test_header_density_in_kg_per_m3():
    overrides, _ = {}, []
    zb._apply_parameter_alias("RHO_MA", 2650.0, "kg/m3", overrides, {}, [])
    assert overrides["rho_ma"] == pytest.approx(2.65)
    geolink = {}
    zb._apply_parameter_alias("RHO_MA", 2650.0, "kg/m3", geolink, {}, [],
                              calculation_profile="geolink")
    assert geolink["rho_ma"] == 2650.0


def test_descriptive_mud_header():
    overrides, unmapped = {}, []
    zb._apply_parameter_alias("MUD", "Water Based", "", overrides, {}, unmapped)
    assert overrides["mud_type"] == "WBM"
    overrides, unmapped = {}, []
    zb._apply_parameter_alias("MUD", "KCl polymer", "", overrides, {}, unmapped)
    assert "mud_type" not in overrides and unmapped == ["MUD"]
    well = pp.read(pp.sample("HW-3"))
    well.set_parameter("MUD", "Water Based")
    result = zb.run_dynamic_workflow_with_zones(well, DynamicWorkflowConfig(rw=0.0682, rw_meas_temp_f=68.0))
    assert result.base_result.config.mud_type == "WBM"


def test_batch_refuses_to_overwrite_sources_and_warns_removed_input_curves(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    shutil.copy(pp.sample("HW-3"), source / "HW-3.las")
    before = (source / "HW-3.las").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        zb.process_las_directory_with_workflow(source, source)
    assert (source / "HW-3.las").read_bytes() == before

    results = zb.process_las_directory_with_workflow(source, tmp_path / "out")
    assert any("PHIE_CC" in message and "kept as" in message for message in results[0].warnings)

    report = zb.write_batch_report(results, tmp_path / "out")
    assert report == tmp_path / "out" / "batch_report.csv"
    audit = zb.write_curve_audit_report(results, tmp_path / "out")
    assert audit == tmp_path / "out" / "curve_audit_report.csv"
    rows = pd.read_csv(audit)
    reason = rows.loc[rows["curve"] == "PHIE_CC", "reason"].iloc[0]
    assert "core permeability" not in str(reason)


def test_json_traces_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        zb.write_workflow_json_traces_directory(tmp_path / "nope", tmp_path / "x")


# --- api -------------------------------------------------------------------------

def test_summary_with_dataframe_setting():
    lithology = pd.read_csv(pp.sample("lithology_30_6-5"))
    result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230,
                          lithology_intervals=lithology)
    summary = result.summary()
    assert "lithology_intervals" in summary["settings_used"]
    json.dumps(summary, default=str)


def test_interpret_rejects_unsupported_input():
    with pytest.raises(TypeError, match="Well"):
        pp.interpret(pd.DataFrame({"DEPTH": [1.0, 2.0]}))


def test_plot_reversed_depth_range_and_docstring_tracks():
    import matplotlib

    matplotlib.use("Agg")
    well = pp.read(pp.sample("30_6-5"))
    viewer = pp.plot(well, depth_range=(3000, 2800), tracks=["GAMMA / HOLE", "RESISTIVITY"])
    try:
        depth_limits = [ax.get_ylim() for ax in viewer.fig.axes if max(ax.get_ylim()) > 1]
        assert depth_limits and all(limits == (3000.0, 2800.0) for limits in depth_limits)
    finally:
        viewer.close()
