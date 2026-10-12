import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from petrophys import Well, sample
from petrophys.interpretation.clay_volume.vcl_module import VclModule
from petrophys.io import read_las
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig


REPORT_ENV_VAR = "WORKFLOW_ROUTING_LOG_DIR"


def _json_safe(value):
    if isinstance(value, np.ndarray):
        data = np.asarray(value)
        if data.ndim == 0:
            return _json_safe(data.item())
        if np.issubdtype(data.dtype, np.number):
            numeric = data.astype(float)
            finite = numeric[np.isfinite(numeric)]
            payload = {
                "type": "ndarray",
                "shape": list(data.shape),
                "finite_count": int(finite.size),
            }
            if finite.size:
                payload.update(
                    {
                        "min": float(np.nanmin(finite)),
                        "max": float(np.nanmax(finite)),
                        "mean": float(np.nanmean(finite)),
                    }
                )
            return payload
        return {
            "type": "ndarray",
            "shape": list(data.shape),
            "sample": [_json_safe(item) for item in data.flat[:5]],
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _curve_summary(well, curve_names):
    summary = {}
    for name in curve_names:
        curve = well.get(name)
        if curve is None:
            continue
        summary[name] = _json_safe(np.asarray(curve.data))
    return summary


def _step_by_name(result, step_name):
    for step in result.steps_executed:
        if getattr(step, "step", None) == step_name:
            return step
    raise AssertionError(f"Workflow step was not executed: {step_name}")


def _step_or_none(result, step_name):
    for step in result.steps_executed:
        if getattr(step, "step", None) == step_name:
            return step
    return None


@pytest.fixture(scope="session")
def workflow_routing_report_paths(tmp_path_factory):
    configured_dir = os.environ.get(REPORT_ENV_VAR)
    report_dir = (
        Path(configured_dir)
        if configured_dir
        else tmp_path_factory.mktemp("workflow_routing_reports")
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_report = report_dir / "routing_cases.jsonl"
    markdown_report = report_dir / "routing_summary.md"
    jsonl_report.unlink(missing_ok=True)
    markdown_report.write_text(
        "# Dynamic Workflow Routing Test Log\n\n"
        "Each section records the workflow route, method, parameters, decisions, "
        "and output curve summaries for one pytest case.\n",
        encoding="utf-8",
    )
    yield jsonl_report, markdown_report
    if configured_dir:
        print(f"\nDynamic workflow routing report: {markdown_report}")


@pytest.fixture
def record_route(workflow_routing_report_paths):
    jsonl_report, markdown_report = workflow_routing_report_paths

    def _record(case_id, result, well, focus_steps=None, curves=None, notes=None):
        focus_steps_local = list(focus_steps or [])
        curves_local = list(curves or [])
        step_payloads = []
        for step in result.steps_executed:
            if focus_steps_local and step.step not in focus_steps_local:
                continue
            step_payloads.append(
                {
                    "step": step.step,
                    "success": step.success,
                    "method": step.method,
                    "inputs": step.input_curves,
                    "outputs": step.output_curves,
                    "parameters": _json_safe(step.parameters),
                    "decisions": list(step.decisions),
                    "message": step.message,
                }
            )

        payload = {
            "case": case_id,
            "workflow_success": result.success,
            "executed_steps": [step.step for step in result.steps_executed],
            "skipped_steps": list(result.steps_skipped),
            "warnings": list(result.warnings),
            "focused_steps": step_payloads,
            "curves": _curve_summary(well, curves_local),
            "notes": notes or "",
        }
        with jsonl_report.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")

        lines = [
            f"\n## {case_id}",
            "",
            f"- workflow_success: `{result.success}`",
            f"- executed_steps: `{', '.join(payload['executed_steps'])}`",
            f"- skipped_steps: `{', '.join(result.steps_skipped) or 'none'}`",
        ]
        if notes:
            lines.append(f"- notes: {notes}")
        if result.warnings:
            lines.append(f"- warnings: `{'; '.join(result.warnings)}`")
        for step in step_payloads:
            lines.extend(
                [
                    "",
                    f"### {step['step']}",
                    f"- success: `{step['success']}`",
                    f"- method: `{step['method']}`",
                    f"- inputs: `{', '.join(step['inputs']) or 'none'}`",
                    f"- outputs: `{', '.join(step['outputs']) or 'none'}`",
                    f"- parameters: `{json.dumps(step['parameters'], sort_keys=True)}`",
                ]
            )
            if step["decisions"]:
                lines.append("- decisions:")
                lines.extend(f"  - {decision}" for decision in step["decisions"])
            if step["message"]:
                lines.append(f"- message: `{step['message']}`")
        if curves_local:
            lines.extend(["", "### Curve Summaries"])
            for name, summary in payload["curves"].items():
                lines.append(f"- {name}: `{json.dumps(summary, sort_keys=True)}`")
        markdown_report.write_text(
            markdown_report.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    return _record


def _routing_well(name="ROUTING-1", n=12):
    depth = np.arange(9000.0, 9000.0 + n, dtype=float)
    clean_to_shale = np.linspace(0.05, 0.95, n)
    well = Well(name)
    well.depth = depth
    well.depth_unit = "ft"
    well.header.td = float(depth[-1])
    well.header.field = "TEST_FIELD"

    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.linspace(35.0, 130.0, n), "API", "Gamma Ray")
    well.add_curve("SP", np.linspace(-85.0, -25.0, n), "mV", "Spontaneous Potential")
    well.add_curve("RHOB", np.linspace(2.18, 2.62, n), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.34, n), "v/v", "Neutron Porosity")
    well.add_curve("DTC", np.linspace(78.0, 125.0, n), "us/ft", "Sonic")
    well.add_curve("PEF", np.linspace(2.0, 5.0, n), "", "PEF")
    well.add_curve("RDEP", np.linspace(90.0, 2.0, n), "ohm.m", "Deep Resistivity")
    well.add_curve("RT", np.linspace(88.0, 2.2, n), "ohm.m", "True Resistivity")
    well.add_curve("RXO", np.linspace(40.0, 1.2, n), "ohm.m", "Flushed Resistivity")
    well.add_curve("RSHA", np.linspace(42.0, 1.4, n), "ohm.m", "Shallow Resistivity")
    well.add_curve("RESS", np.linspace(40.0, 1.2, n), "ohm.m", "Shallow Resistivity")
    well.add_curve("CALI", np.array([8.5, 8.6, 8.4, 10.4, 10.6, 8.5, 8.5, 8.6, 8.4, 8.5, 8.5, 8.6])[:n], "in", "Caliper")
    well.add_curve("CAL", np.array([8.5, 8.6, 8.4, 10.4, 10.6, 8.5, 8.5, 8.6, 8.4, 8.5, 8.5, 8.6])[:n], "in", "Caliper")
    well.add_curve("BS", np.full(n, 8.5), "in", "Bit Size")
    well.add_curve("DRHO", np.array([0.01, 0.02, 0.01, 0.17, 0.18, 0.01, 0.0, 0.01, 0.0, 0.02, 0.01, 0.0])[:n], "g/cc", "Density Correction")
    well.add_curve("VCL", clean_to_shale, "v/v", "Clay Volume")
    well.add_curve("VCLAY", clean_to_shale, "v/v", "Clay Volume")
    well.add_curve("PHIT", np.linspace(0.24, 0.10, n), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.linspace(0.22, 0.06, n), "v/v", "Effective Porosity")
    well.add_curve("SW", np.linspace(0.25, 0.90, n), "v/v", "Water Saturation")
    well.add_curve("SXO", np.linspace(0.55, 1.00, n), "v/v", "Flushed Saturation")
    well.add_curve("PERM", np.linspace(250.0, 0.5, n), "mD", "Permeability")
    return well


def _remove_curves(well, *names):
    for name in names:
        well.remove_curve(name)
    return well


def _expected_adaptive_rwa(well, result, step):
    """Reconstruct the adaptive Rwa clean-wet Rw from the step's reported params.

    Mirrors the workflow: clean-wet mask at the reported cutoffs (the tight cut
    also excludes bad hole), direct Arps temperature correction, then the
    reported adaptive percentile (P50 for a narrow all-wet spread, P10 when a
    hydrocarbon leg broadens it).
    """
    rdep = np.asarray(well.get("RDEP").data, dtype=float)
    phie = np.asarray(well.get("PHIE").data, dtype=float)
    vcl = np.asarray(well.get("VCL").data, dtype=float)
    a = step.parameters["archie_a"]
    m = step.parameters["archie_m"]
    vcut = step.parameters["vcl_cutoff"]
    pcut = step.parameters["phie_cutoff"]
    rwa = rdep * np.power(phie, m) / a
    valid = (np.isfinite(rdep) & np.isfinite(phie) & np.isfinite(vcl)
             & np.isfinite(rwa) & (rwa > 0) & (vcl < vcut) & (phie > pcut))
    bad_hole = result.well.get("BAD_HOLE")
    if vcut <= 0.2 and bad_hole is not None:
        valid &= ~(np.asarray(bad_hole.data, dtype=float) > 0.5)
    temp = np.asarray(result.well.get("TEMP").data, dtype=float)
    ref = step.parameters["rwa_reference_temp_f"]
    rwa_ref = rwa[valid] * (temp[valid] + 6.77) / (ref + 6.77)
    return float(np.percentile(rwa_ref, step.parameters["rwa_percentile"]))


def _run(case_id, well, config=None, record_route=None, focus_steps=None, curves=None, notes=None):
    result = DynamicWorkflow(config or DynamicWorkflowConfig()).run(well)
    if record_route is not None:
        record_route(case_id, result, well, focus_steps=focus_steps, curves=curves, notes=notes)
    return result


def test_full_dynamic_workflow_records_trace_for_every_module(record_route):
    well = _routing_well()
    config = DynamicWorkflowConfig(
        bht_f=220.0,
        td_ft=float(well.depth[-1]),
        rmf_at_meas_temp=0.2,
        rmf_meas_temp_f=75.0,
        rw=0.05,
        rw_meas_temp_f=75.0,
        sw_model="auto",
        salinity_ppm=35000.0,
        perm_method="timur",
        swirr=0.25,
        lithology="clastic",
        apply_hc_correction=True,
    )

    result = _run(
        "full_pipeline_all_modules",
        well,
        config,
        record_route,
        focus_steps=[step_enum.value for step_enum, _, _ in DynamicWorkflow().get_pipeline()],
        curves=["TEMP", "VCL", "RCL", "RW", "RMF", "SW", "SXO", "PERM", "BVW", "MHI", "NET_PAY"],
    )

    expected_steps = {
        "temperature", "bad_hole", "vclay", "rcl", "lithology", "rmf", "rw",
        "invasion", "hc_correction", "porosity",
        "saturation", "permeability", "rock_quality", "bvw", "mhi", "net_pay",
    }
    executed = {step.step for step in result.steps_executed if step.success}
    assert expected_steps.issubset(executed)
    assert result.success


def test_pipeline_solves_provisional_porosity_before_lithology_then_refines():
    pipeline_steps = [step.value for step, _, _ in DynamicWorkflow().get_pipeline()]

    assert pipeline_steps.index("porosity") < pipeline_steps.index("lithology")
    assert pipeline_steps.index("lithology") < pipeline_steps.index("porosity_refinement")
    assert pipeline_steps.index("porosity_refinement") < pipeline_steps.index("core_correction")
    assert pipeline_steps.index("core_correction") < pipeline_steps.index("saturation")


def test_workflow_recalculates_interpreted_vcl_and_phie():
    depth = np.arange(1000.0, 1006.0)
    well = Well("VENDOR-INTERP")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("GR", np.array([45.0, 50.0, 55.0, 60.0, 65.0, 70.0]), "API", "Gamma Ray")
    well.add_curve("RHOB", np.array([2.45, 2.43, 2.42, 2.40, 2.39, 2.38]), "g/cm3", "Bulk Density")
    well.add_curve("NPHI", np.array([0.18, 0.17, 0.16, 0.15, 0.14, 0.13]), "v/v", "Neutron")
    well.add_curve("PEF", np.full(depth.size, 2.8), "", "Photoelectric Factor")
    well.add_curve("RDEP", np.full(depth.size, 20.0), "ohm.m", "Deep Resistivity")
    well.add_curve("RSHA", np.full(depth.size, 10.0), "ohm.m", "Shallow Resistivity")
    well.add_curve("BS", np.full(depth.size, 8.5), "in", "Bit Size")

    vendor_vcl = np.linspace(0.72, 0.95, depth.size)
    vendor_phie = np.linspace(0.02, 0.05, depth.size)
    well.add_curve(
        "VCL",
        vendor_vcl,
        "v/v",
        "Volume of Clay type Material relative to Total Volume from HILT Lithology Computation",
    )
    well["VCL"].metadata.source = "DLIS"
    well.add_curve(
        "PHIE",
        vendor_phie,
        "v/v",
        "Effective Porosity from HILT Lithology Computation",
    )
    well["PHIE"].metadata.source = "DLIS"

    result = DynamicWorkflow(
        DynamicWorkflowConfig(rw=0.05, lithology="clastic", matrix_type="sandstone")
    ).run(well)

    assert result.success
    assert not np.allclose(well["VCL"].data, vendor_vcl)
    assert not np.allclose(well["PHIE"].data, vendor_phie)
    assert "VCL_CALC" not in well
    assert "PHIE_CALC" not in well

    vcl_step = _step_by_name(result, "vclay")
    assert vcl_step.parameters["active_vcl_source"] == "workflow_calculation"
    assert vcl_step.parameters["computed_vcl_curve"] == "VCL"


def test_dynamic_workflow_skips_modules_when_inputs_are_missing(record_route):
    well = Well("DEPTH-ONLY")
    depth = np.arange(1000.0, 1005.0)
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")

    result = _run(
        "depth_only_skip_map",
        well,
        DynamicWorkflowConfig(),
        record_route,
        focus_steps=["temperature", "rw"],
        curves=["TEMP", "RW"],
    )

    assert result.success is False
    assert result.status == "failed"
    assert result.steps_executed == []
    assert result.errors == ["Well has no finite log-curve samples"]
    assert "TEMP" not in well
    assert "RW" not in well


@pytest.mark.parametrize(
    "case_id,bht_value,bht_unit,expected_reference,expected_method",
    [
        ("temperature_bht_f", 212.0, "F", 212.0, "linear_gradient_from_BHT"),
        ("temperature_bht_c_unit_corrected", 100.0, "C", 212.0, "linear_gradient_from_BHT"),
        ("temperature_bht_missing_unit_treated_as_f", 100.0, "", 100.0, "linear_gradient_from_BHT"),
    ],
)
def test_temperature_routing_and_units(
    case_id, bht_value, bht_unit, expected_reference, expected_method, record_route
):
    well = _routing_well(case_id)
    well.header.bht = bht_value
    well.header.td = float(well.depth[-1])
    if bht_unit:
        well.set_parameter("BHT__UNIT", bht_unit)

    result = _run(
        case_id,
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["temperature"],
        curves=["TEMP"],
    )
    step = _step_by_name(result, "temperature")
    assert step.method == expected_method
    assert np.isclose(step.parameters["reference_temperature_f"], expected_reference)


def test_temperature_ignores_fmt_and_uses_fallback_without_bht(record_route):
    well = _routing_well("TEMP-FMT-IGNORED")
    well.header.bht = None
    well.set_parameter("FMT", 80.0)
    well.set_parameter("FMT__UNIT", "C")
    well.set_parameter("FMTD", float(well.depth[-1]))
    well.set_parameter("FMTD__UNIT", "ft")

    result = _run(
        "temperature_fmt_ignored_fallback",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["temperature"],
        curves=["TEMP"],
    )
    assert _step_by_name(result, "temperature").method == "fallback_geothermal_gradient"


@pytest.mark.parametrize(
    "case_id,mutate,config,expected_sp,expected_nd",
    [
        ("vclay_gr_sp_nd_enabled", lambda well: well, DynamicWorkflowConfig(calculation_profile="benchmark"), True, True),
        ("vclay_sp_disabled_by_kcl_mud", lambda well: well, DynamicWorkflowConfig(mud_has_clay_or_kcl=True, calculation_profile="benchmark"), False, True),
        ("vclay_nd_disabled_when_missing_density_neutron", lambda well: _remove_curves(well, "NPHI", "RHOB"), DynamicWorkflowConfig(calculation_profile="benchmark"), True, False),
    ],
)
def test_vclay_routing_by_available_methods(
    case_id, mutate, config, expected_sp, expected_nd, record_route
):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, config, record_route, focus_steps=["vclay"], curves=["VCL", "VCLGR", "VCLSP", "VCLND"])
    step = _step_by_name(result, "vclay")
    assert step.parameters["use_sp"] is expected_sp
    assert step.parameters["use_nd"] is expected_nd
    assert "VCLAY" in step.output_curves
    method_curves = [well.get(name) for name in ("VCLGR", "VCLSP", "VCLND")]
    method_data = [curve.data for curve in method_curves if curve is not None]
    stacked = np.vstack(method_data)
    expected = np.full(well.n_samples, np.nan)
    valid = np.any(np.isfinite(stacked), axis=0)
    expected[valid] = np.nanmin(stacked[:, valid], axis=0)
    np.testing.assert_allclose(well.get("VCL").data, expected, equal_nan=True)
    assert any("min(available methods)" in decision for decision in step.decisions)


def test_vclay_default_keeps_whole_log_sp_out_of_the_minimum(record_route):
    well = _routing_well("VCL-CURRENT-SP")
    result = _run(
        "vclay_current_sp_reported", well, DynamicWorkflowConfig(), record_route,
        focus_steps=["vclay"], curves=["VCL", "VCLGR", "VCLSP", "VCLND"],
    )
    step = _step_by_name(result, "vclay")

    assert step.parameters["use_sp"] is True and step.parameters["sp_in_vcl"] is False
    expected = np.fmin(well.get("VCLGR").data, well.get("VCLND").data)
    np.testing.assert_allclose(well.get("VCL").data, expected)
    assert well.get("VCLSP") is not None
    assert any(decision.startswith("Final VCL = min(GR, ND)") for decision in step.decisions)


def test_vclay_keeps_nan_where_all_methods_are_missing_without_runtime_warning():
    well = Well("VCL-ALL-MISSING-ROW")
    well.depth = np.array([1000.0, 1001.0, 1002.0])
    well.add_curve("GR", np.array([40.0, np.nan, 90.0]), "API", "Gamma Ray")
    well.add_curve("NPHI", np.array([0.15, np.nan, 0.35]), "v/v", "Neutron")
    well.add_curve("RHOB", np.array([2.45, np.nan, 2.20]), "g/cc", "Density")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = VclModule(gr_min=20.0, gr_max=120.0, use_sp=False, use_nd=True).calculate(well)

    assert np.isnan(result.vcl[1])
    assert not any(
        issubclass(item.category, RuntimeWarning) and "All-NaN slice" in str(item.message)
        for item in caught
    )


def test_vclay_trace_reports_the_canonical_clean_line(record_route):
    well = _routing_well("VCL-SANDSTONE-CLEAN-LINE")
    result = _run(
        "vclay_sandstone_clean_line",
        well,
        DynamicWorkflowConfig(
            matrix_type="sandstone", lithology="sandstone", calculation_profile="benchmark"
        ),
        record_route,
        focus_steps=["vclay"],
        curves=["VCL", "VCLND"],
    )
    parameters = _step_by_name(result, "vclay").parameters

    assert parameters["nd_matrix"] == "limestone"
    assert parameters["nd_clean1_density"] == pytest.approx(2.71)
    assert parameters["nd_clean1_neutron"] == pytest.approx(0.0)
    assert parameters["nd_clean2_density"] == pytest.approx(2.03)
    assert parameters["nd_clean2_neutron"] == pytest.approx(0.40)


def test_vclay_trace_reports_the_clean_line_of_the_matrix(record_route):
    well = _routing_well("VCL-MATRIX-CLEAN-LINE")
    result = _run(
        "vclay_matrix_clean_line",
        well,
        DynamicWorkflowConfig(matrix_type="sandstone", lithology="sandstone"),
        record_route,
        focus_steps=["vclay"],
        curves=["VCL", "VCLND"],
    )
    parameters = _step_by_name(result, "vclay").parameters

    assert parameters["nd_matrix"] == "sandstone"
    assert parameters["nd_clean1_density"] == pytest.approx(2.65)
    assert parameters["nd_clean1_neutron"] == pytest.approx(-0.02)
    assert parameters["nd_clean2_density"] == pytest.approx(1.99)
    assert parameters["nd_clean2_neutron"] == pytest.approx(0.38)


@pytest.mark.parametrize("curve_to_keep", ["CALI", "CAL"])
def test_bad_hole_accepts_cali_and_cal_namings(curve_to_keep, record_route):
    case_id = f"bad_hole_{curve_to_keep.lower()}_only"
    well = _routing_well(case_id)
    remove_name = "CAL" if curve_to_keep == "CALI" else "CALI"
    _remove_curves(well, remove_name, "DRHO")

    result = _run(
        case_id,
        well,
        DynamicWorkflowConfig(skip_steps=["vclay", "lithology", "porosity", "saturation"]),
        record_route,
        focus_steps=["bad_hole"],
        curves=["BAD_HOLE"],
    )
    step = _step_by_name(result, "bad_hole")

    assert curve_to_keep in step.input_curves
    assert well.get("BAD_HOLE") is not None
    assert np.nansum(well["BAD_HOLE"].data) == 2


def test_bad_hole_uses_depth_dependent_bs_curve(record_route):
    well = _routing_well("BAD-HOLE-DEPTH-BS")
    well.add_curve(
        "BS",
        np.array([8.5, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25, 12.25]),
        "in",
        "Bit Size",
    )
    well.add_curve(
        "CALI",
        np.array([10.1, 13.0, 13.2, 13.1, 13.0, 12.8, 12.7, 12.8, 12.6, 12.9, 12.7, 12.8]),
        "in",
        "Caliper",
    )
    well.add_curve("DRHO", np.zeros(well.n_samples), "g/cc", "Density Correction")

    result = _run(
        "bad_hole_depth_dependent_bs",
        well,
        DynamicWorkflowConfig(skip_steps=["vclay", "lithology", "porosity", "saturation"]),
        record_route,
        focus_steps=["bad_hole"],
        curves=["BAD_HOLE"],
    )
    step = _step_by_name(result, "bad_hole")

    assert step.parameters["bit_size_source"] == "BS_curve_depth_by_depth"
    assert step.parameters["bit_size_depth_dependent"] is True
    assert step.parameters["bit_size_used"] == 12.25
    np.testing.assert_array_equal(
        well["BAD_HOLE"].data.astype(bool),
        np.array([True, False, False, False, False, False, False, False, False, False, False, False]),
    )


@pytest.mark.parametrize(
    "profile, bit_size_used, bit_size_source",
    [("current", None, "none"), ("benchmark", 8.5, "fallback_8.5")],
)
def test_bad_hole_without_bs(profile, bit_size_used, bit_size_source, record_route):
    well = _routing_well("BAD-HOLE-NO-BS")
    _remove_curves(well, "BS")
    well.header.bit_size = None

    result = _run(
        f"bad_hole_no_synthetic_bs_{profile}",
        well,
        DynamicWorkflowConfig(
            skip_steps=["vclay", "lithology", "porosity", "saturation"],
            calculation_profile=profile,
        ),
        record_route,
        focus_steps=["bad_hole"],
        curves=["BAD_HOLE"],
    )
    step = _step_by_name(result, "bad_hole")

    # "current" assumes no bit size and flags |DRHO| > 0.15 g/cm3 alone; the
    # benchmark tests the caliper against an assumed 8.5 in bit. Both flag
    # the 10.4-10.6 in washout, where DRHO reads 0.17-0.18 g/cm3.
    assert "BS" not in step.input_curves
    assert step.parameters["bit_size_used"] == bit_size_used
    assert step.parameters["bit_size_source"] == bit_size_source
    assert well["BAD_HOLE"].data.astype(int).tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "case_id,mutate,expected_method",
    [
        ("rcl_from_vcl_high_shale", lambda well: well, "vcl_gt_0.9_median_rt"),
        (
            "rcl_from_lithology_shale",
            lambda well: (
                well.add_curve("LITHOLOGY_GEOLINK", np.array([1, 1, 7, 7, 9, 9, 1, 1, 7, 7, 1, 1]), "", "Lithology")
                and well
            ),
            "lithology_shale_median_rt",
        ),
        ("rcl_default_without_shale_samples", lambda well: (well.add_curve("VCLAY", np.full(well.n_samples, 0.2), "v/v", "Clay Volume") and well), "default"),
    ],
)
def test_rcl_routing_sources(case_id, mutate, expected_method, record_route):
    well = mutate(_routing_well(case_id))
    if case_id == "rcl_default_without_shale_samples":
        well.add_curve("VCL", np.full(well.n_samples, 0.2), "v/v", "Clay Volume")
        well.add_curve("VCLAY", np.full(well.n_samples, 0.2), "v/v", "Clay Volume")
    result = _run(
        case_id,
        well,
        DynamicWorkflowConfig(rw=0.05, skip_steps=["vclay"]),
        record_route,
        focus_steps=["rcl"],
        curves=["VCL", "VCLAY", "RDEP"],
    )
    step = _step_by_name(result, "rcl")
    assert step.method == expected_method
    assert "rcl_computed" in step.parameters


def test_rcl_uses_vcl_when_vclay_alias_is_missing(record_route):
    well = _remove_curves(_routing_well("RCL-VCL-ONLY"), "VCLAY")
    result = _run(
        "rcl_vcl_only",
        well,
        DynamicWorkflowConfig(rw=0.05, skip_steps=["vclay"]),
        record_route,
        focus_steps=["rcl"],
        curves=["VCL", "VCLAY"],
    )
    step = _step_by_name(result, "rcl")

    assert step.method == "vcl_gt_0.9_median_rt"
    assert step.input_curves == ["RDEP", "VCL"]


def test_lithology_geolink_keeps_input_lithology_but_solves_volumes(record_route):
    well = _routing_well("lithology_geolink_authoritative", n=6)
    well.add_curve("LITHOLOGY_GEOLINK", np.array([1, 1, 9, 15, 7, 5]), "", "Lithology")

    result = _run(
        "lithology_geolink_authoritative",
        well,
        DynamicWorkflowConfig(
            skip_steps=[
                "temperature", "bad_hole", "vclay", "rcl", "rmf", "rw", "invasion",
                "hc_correction", "porosity", "saturation", "permeability",
                "rock_quality", "bvw", "mhi", "net_pay",
            ]
        ),
        record_route,
        focus_steps=["lithology"],
        curves=["LITHOLOGY_GEOLINK", "VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"],
    )
    step = _step_by_name(result, "lithology")

    assert step.parameters["lithology_curve_written"] is False
    assert "LITHO" not in step.output_curves
    assert well.get("LITHO") is None
    for curve in ("VQUAR", "VCALC", "VDOLO", "VANHY"):
        assert well.get(curve) is not None


def test_lithology_fallback_writes_litho_when_input_lithology_absent(record_route):
    well = _routing_well("lithology_fallback", n=6)

    result = _run(
        "lithology_fallback",
        well,
        DynamicWorkflowConfig(
            skip_steps=[
                "temperature", "bad_hole", "vclay", "rcl", "rmf", "rw", "invasion",
                "hc_correction", "porosity", "saturation", "permeability",
                "rock_quality", "bvw", "mhi", "net_pay",
            ]
        ),
        record_route,
        focus_steps=["lithology"],
        curves=["VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"],
    )
    step = _step_by_name(result, "lithology")

    assert step.parameters["lithology_curve_written"] is True
    assert "LITHO" in step.output_curves
    for curve in ("VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"):
        assert well.get(curve) is not None


def test_canonical_lithology_requires_neutron_even_with_pef_dtc(record_route):
    well = _remove_curves(_routing_well("lithology_without_neutron", n=8), "NPHI", "PHIN")

    result = _run(
        "lithology_without_neutron",
        well,
        DynamicWorkflowConfig(
            rw=0.05,
            skip_steps=[
                "temperature", "bad_hole", "rcl", "rmf", "hc_correction",
                "porosity_refinement", "core_correction", "invasion", "saturation",
                "permeability", "rock_quality", "bvw", "mhi", "net_pay",
            ],
        ),
        record_route,
        focus_steps=["porosity", "lithology"],
        curves=["PHID", "PHIT", "PHIE", "VQUAR", "VCALC", "VDOLO", "VANHY", "MMQC", "LITHO"],
    )

    assert _step_or_none(result, "lithology") is None
    assert "lithology" in result.steps_skipped
    for curve in ("VQUAR", "VCALC", "VDOLO", "VANHY"):
        assert well.get(curve) is None
    assert well.get("MMQC") is None


@pytest.mark.parametrize(
    "case_id,mutate,config,expected_method,expected_temp_key,expected_temp_value",
    [
        ("rmf_obm_default", lambda well: well, DynamicWorkflowConfig(mud_type="OBM"), "OBM_default", None, None),
        (
            "rmf_las_rmf_temp_c_corrected",
            lambda well: (well.set_parameter("RMF", 0.20), well.set_parameter("RMFT", 25.0), well.set_parameter("RMFT__UNIT", "C"), well)[-1],
            DynamicWorkflowConfig(),
            "las_metadata_profile_temperature_correction",
            "measured_temp_f",
            77.0,
        ),
        (
            "rmf_rm_to_rmf_temp_c_corrected",
            lambda well: (well.set_parameter("RM", 0.60), well.set_parameter("RMT", 25.0), well.set_parameter("RMT__UNIT", "C"), well)[-1],
            DynamicWorkflowConfig(),
            "rm_to_rmf_conversion_factor_0.75",
            "rm_measured_temp_f",
            77.0,
        ),
        (
            "rmf_from_mud_filtrate_salinity",
            lambda well: well,
            DynamicWorkflowConfig(rmf_salinity_ppm=90000.0, rmf_meas_temp_f=75.0),
            "mud_filtrate_salinity_conversion",
            "measured_temp_f",
            75.0,
        ),
    ],
)
def test_rmf_routing_sources_and_temperature_units(
    case_id, mutate, config, expected_method, expected_temp_key, expected_temp_value, record_route
):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, config, record_route, focus_steps=["temperature", "rmf"], curves=["TEMP", "RM", "RMF"])
    step = _step_by_name(result, "rmf")
    assert step.method == expected_method
    if expected_temp_key is not None:
        assert np.isclose(step.parameters[expected_temp_key], expected_temp_value)


@pytest.mark.parametrize(
    "case_id,mutate,config,expected_method",
    [
        (
            "rw_config_with_measured_temp",
            lambda well: well,
            DynamicWorkflowConfig(rw=0.05, rw_meas_temp_f=75.0),
            "user_provided_temperature_profile_corrected",
        ),
        (
            "rw_config_default_reference_temp",
            lambda well: well,
            DynamicWorkflowConfig(rw=0.05),
            "user_provided_assumed_reference_temperature",
        ),
        (
            "rw_las_rw_temp_c_corrected",
            lambda well: (well.set_parameter("RW", 0.05), well.set_parameter("RWT", 25.0), well.set_parameter("RWT__UNIT", "C"), well)[-1],
            DynamicWorkflowConfig(),
            "las_metadata_temperature_profile_corrected",
        ),
        (
            "rw_las_rw_default_reference_temp",
            lambda well: (well.set_parameter("RW", 0.05), well)[-1],
            DynamicWorkflowConfig(),
            "las_metadata_assumed_reference_temperature",
        ),
        (
            "rw_from_salinity",
            lambda well: well,
            DynamicWorkflowConfig(salinity_ppm=35000.0),
            "salinity_conversion",
        ),
        (
            "rw_rwa_clean_wet_fallback",
            lambda well: _remove_curves(well, "SP", "RXO", "RSHA", "RESS"),
            DynamicWorkflowConfig(),
            "rwa_clean_wet_p5",
        ),
    ],
)
def test_rw_routing_sources_and_temperature_units(case_id, mutate, config, expected_method, record_route):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, config, record_route, focus_steps=["temperature", "rmf", "rw"], curves=["TEMP", "RW", "RMF"])
    step = _step_by_name(result, "rw")
    assert step.method == expected_method
    if case_id == "rw_las_rw_temp_c_corrected":
        assert np.isclose(step.parameters["measured_temp_f"], 77.0)
    if expected_method in {
        "user_provided_temperature_profile_corrected",
        "las_metadata_temperature_profile_corrected",
    }:
        rw_in = step.parameters["rw_at_measured_temp"]
        measured_temp = step.parameters["measured_temp_f"]
        expected_min = rw_in * (measured_temp + 6.77) / (step.parameters["formation_temp_max_f"] + 6.77)
        expected_max = rw_in * (measured_temp + 6.77) / (step.parameters["formation_temp_min_f"] + 6.77)
        assert np.isclose(step.parameters["rw_min"], expected_min)
        assert np.isclose(step.parameters["rw_max"], expected_max)
    if expected_method in {
        "user_provided_assumed_reference_temperature",
        "las_metadata_assumed_reference_temperature",
    }:
        # Without a measurement temperature, Rw is taken at the median formation temperature.
        reference = step.parameters["reference_temp_f"]
        assert step.parameters["formation_temp_min_f"] < reference < step.parameters["formation_temp_max_f"]
        rw_in = step.parameters["rw"]
        expected_min = rw_in * (reference + 6.77) / (step.parameters["formation_temp_max_f"] + 6.77)
        expected_max = rw_in * (reference + 6.77) / (step.parameters["formation_temp_min_f"] + 6.77)
        assert np.isclose(step.parameters["rw_min"], expected_min)
        assert np.isclose(step.parameters["rw_max"], expected_max)
    if case_id == "rw_rwa_clean_wet_fallback":
        assert "rwa_reference_temp_f" in step.parameters
        assert step.parameters["rw_min"] < step.parameters["rw_max"]
        assert step.parameters["rwa_percentile"] == 5.0
        assert np.isfinite(step.parameters["rw_from_rwa"])


def test_rw_rwa_fallback_uses_lithology_specific_archie_coefficients(record_route):
    well = _remove_curves(_routing_well("rw_rwa_limestone_archie"), "SP", "RXO", "RSHA", "RESS")
    well.add_curve("LITHOLOGY_GEOLINK", np.full(well.n_samples, 11.0), "", "Lithology")

    result = _run(
        "rw_rwa_limestone_archie",
        well,
        DynamicWorkflowConfig(),
        record_route,
        focus_steps=["rw"],
        curves=["RW"],
    )
    step = _step_by_name(result, "rw")

    assert step.method == "rwa_clean_wet_p5"
    assert step.parameters["archie_coefficients"]["LIMESTONE"]["a"] == 1.0
    assert step.parameters["archie_a"] == 1.0
    assert step.parameters["rwa_percentile"] == 5.0


def test_saturation_uses_lithology_specific_archie_coefficients(record_route):
    well = _routing_well("sw_limestone_archie")
    well.add_curve("LITHOLOGY_GEOLINK", np.full(well.n_samples, 11.0), "", "Lithology")
    well.get("VCL").data[:] = 0.05
    well.get("VCLAY").data[:] = 0.05

    result = _run(
        "sw_limestone_archie",
        well,
        # Dual Water runs only with a supplied Rwb.
        DynamicWorkflowConfig(rw=0.05, rwb=0.02),
        record_route,
        focus_steps=["saturation"],
        curves=["SW"],
    )
    step = _step_by_name(result, "saturation")

    assert step.parameters["archie_coefficients"]["LIMESTONE"]["a"] == 1.0
    assert step.parameters["a"] == 1.0
    assert step.parameters["m"] == 2.0
    assert step.parameters["n"] == 2.0
    assert set(step.parameters["sw_model_sample_counts"]).issubset({"archie", "dual_water"})


def test_invasion_routes_when_shallow_and_deep_resistivity_exist(record_route):
    well = _routing_well("INVASION-PRESENT")
    result = _run("invasion_present", well, DynamicWorkflowConfig(rw=0.05), record_route, focus_steps=["invasion"], curves=["INVASION_RATIO"])
    assert _step_by_name(result, "invasion").method == "rxo_rt_ratio"
    assert well.get("INVASION_RATIO") is not None


def test_invasion_skips_without_shallow_resistivity(record_route):
    well = _remove_curves(_routing_well("INVASION-MISSING"), "RXO", "RSHA", "RESS")
    result = _run("invasion_missing_shallow", well, DynamicWorkflowConfig(rw=0.05), record_route, focus_steps=["invasion"])
    assert _step_or_none(result, "invasion") is None
    assert "invasion" in result.steps_skipped


def test_invasion_falls_back_to_medium_deep_proxy(record_route):
    well = _remove_curves(_routing_well("invasion_rmed_proxy"), "RXO", "RSHA", "RESS")
    well.add_curve("RMED", np.linspace(30.0, 1.0, well.n_samples), "ohm.m", "Medium Resistivity")

    result = _run(
        "invasion_rmed_proxy",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["invasion"],
        curves=["INVASION_RATIO"],
    )
    step = _step_by_name(result, "invasion")

    assert step.parameters["rxo_curve"] == "RMED"
    assert step.parameters["fallback_proxy_used"] is True
    assert step.parameters["medium_deep_proxy_samples"] == well.n_samples
    assert well.get("INVASION_RATIO") is not None


@pytest.mark.parametrize(
    "case_id,keep_curve",
    [
        ("invasion_rsha_only", "RSHA"),
        ("invasion_ress_only", "RESS"),
    ],
)
def test_invasion_accepts_rsha_ress_or_rxo_shallow_namings(case_id, keep_curve, record_route):
    well = _routing_well(case_id)
    for curve in {"RSHA", "RESS", "RXO"} - {keep_curve}:
        well.remove_curve(curve)

    result = _run(
        case_id,
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["invasion"],
        curves=["INVASION_RATIO"],
    )
    step = _step_by_name(result, "invasion")

    assert step.input_curves == ["RDEP", keep_curve]
    assert well.get("INVASION_RATIO") is not None


def test_invasion_uses_finite_fallback_alias_when_primary_shallow_is_nonfinite(record_route):
    well = _routing_well("invasion_nonfinite_primary")
    well.get("RSHA").data[:] = np.nan
    well.remove_curve("RESS")

    result = _run(
        "invasion_nonfinite_primary",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["invasion"],
        curves=["INVASION_RATIO"],
    )
    step = _step_by_name(result, "invasion")

    assert step.input_curves == ["RDEP", "RXO"]
    assert step.parameters["rxo_curve"] == "RXO"
    assert well.get("INVASION_RATIO") is not None


def test_invasion_combines_partial_shallow_aliases_depthwise(record_route):
    well = _routing_well("invasion_partial_aliases")
    well.get("RSHA").data[:] = np.nan
    well.get("RSHA").data[::2] = 8.0
    well.get("RXO").data[:] = np.nan
    well.get("RXO").data[1::2] = 9.0
    well.remove_curve("RESS")

    result = _run(
        "invasion_partial_aliases",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["invasion"],
        curves=["INVASION_RATIO"],
    )
    step = _step_by_name(result, "invasion")

    # The flushed-zone RXO comes first; RSHA fills the samples where it is missing.
    assert step.input_curves == ["RDEP", "RXO+RSHA"]
    assert step.parameters["rxo_curve"] == "RXO+RSHA"
    assert step.parameters["rxo_source_counts"] == {"RXO": 6, "RSHA": 6}
    assert step.parameters["valid_samples"] == well.n_samples
    assert np.isfinite(well.get("INVASION_RATIO").data).all()


@pytest.mark.parametrize("profile", ["current", "benchmark"])
def test_invasion_reads_the_flushed_zone_first_under_current(profile, record_route):
    # RXO 40 -> 1.2, RSHA 42 -> 1.4 and RDEP 90 -> 2 ohm.m over 12 samples; RXO is missing at the third sample.
    well = _routing_well(f"invasion_order_{profile}")
    well.remove_curve("RESS")
    well.get("RXO").data[2] = np.nan
    rxo, rsha, rdep = (np.asarray(well.get(name).data, dtype=float).copy() for name in ("RXO", "RSHA", "RDEP"))

    result = _run(
        f"invasion_order_{profile}",
        well,
        DynamicWorkflowConfig(rw=0.05, calculation_profile=profile),
        record_route,
        focus_steps=["invasion"],
        curves=["INVASION_RATIO"],
    )
    step = _step_by_name(result, "invasion")
    ratio = np.asarray(well.get("INVASION_RATIO").data, dtype=float)

    if profile == "current":
        np.testing.assert_allclose(ratio[[0, 2, 11]], [40.0 / 90.0, rsha[2] / rdep[2], 1.2 / 2.0])
        assert step.parameters["rxo_source_counts"] == {"RXO": 11, "RSHA": 1}
        assert step.parameters["rxo_curve_order"] == ["RXO", "RSHA", "RESS", "RMED", "RESM"]
        assert step.decisions[0] == (
            "INVASION_RATIO reads RXO at 11 and RSHA at 1 samples over RDEP; curve preference RXO, RSHA, RESS, "
            "then the RMED/RESM medium/deep proxy"
        )
    else:
        # The eval package read RSHA first; the trace records nothing new.
        np.testing.assert_allclose(ratio, rsha / rdep)
        assert step.parameters["rxo_curve"] == "RSHA"
        assert "rxo_curve_order" not in step.parameters
        assert not any(decision.startswith("INVASION_RATIO reads") for decision in step.decisions)
    assert np.isnan(rxo[2])


def test_mhi_removes_existing_curve_when_sxo_is_only_quicklook_fallback(record_route):
    well = _remove_curves(_routing_well("mhi_existing_quicklook_only"), "RXO", "RSHA", "RESS")
    well.add_curve("MHI", np.linspace(0.1, 0.9, well.n_samples), "", "Old MHI")

    result = _run(
        "mhi_existing_quicklook_only",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["saturation", "mhi"],
        curves=["SXO", "MHI"],
    )
    step = _step_by_name(result, "mhi")

    assert step.output_curves == []
    assert step.parameters["diagnostic_sxo_samples"] == 0
    assert well.get("MHI") is None


@pytest.mark.parametrize(
    "case_id,config,expected_rho_ma",
    [
        ("hc_matrix_from_sandstone_defaults", DynamicWorkflowConfig(matrix_type="sandstone", apply_hc_correction=True), 2.65),
        ("hc_user_matrix_density_override", DynamicWorkflowConfig(matrix_type="sandstone", rho_ma=2.71, apply_hc_correction=True), 2.71),
    ],
)
def test_hc_correction_matrix_routing(case_id, config, expected_rho_ma, record_route):
    well = _routing_well(case_id)
    result = _run(case_id, well, config, record_route, focus_steps=["hc_correction"], curves=["NPHI_HC", "RHOB_HC"])
    step = _step_by_name(result, "hc_correction")
    assert step.parameters["rho_ma"] == expected_rho_ma
    assert set(step.output_curves) == {"NPHI_HC", "RHOB_HC"}


def test_hc_correction_is_opt_in_by_default(record_route):
    well = _routing_well("hc_correction_default_off")
    result = _run(
        "hc_correction_default_off",
        well,
        DynamicWorkflowConfig(),
        record_route,
        focus_steps=["hc_correction"],
        curves=["NPHI_HC", "RHOB_HC"],
    )

    assert "hc_correction" in result.steps_skipped
    assert well.get("NPHI_HC") is None
    assert well.get("RHOB_HC") is None


@pytest.mark.parametrize(
    "case_id,mutate,expected_method,expected_apparent_source",
    [
        ("porosity_full_density_neutron_sonic", lambda well: well, "adaptive_nd", None),
        ("porosity_density_only", lambda well: _remove_curves(well, "NPHI", "DTC", "PHIN", "PHIS", "PHIT"), "density_only", "density-derived"),
        ("porosity_neutron_only", lambda well: _remove_curves(well, "RHOB", "DTC", "PHID", "PHIS", "PHIT"), "neutron_only", "neutron-derived"),
        ("porosity_sonic_only", lambda well: _remove_curves(well, "NPHI", "RHOB", "PHIN", "PHID", "PHIT"), "sonic_only", "sonic-derived"),
        ("porosity_existing_phit_reused", lambda well: _remove_curves(well, "NPHI", "RHOB", "DTC", "PHIN", "PHID", "PHIS"), "existing_phit", None),
    ],
)
def test_porosity_routes_by_available_inputs(case_id, mutate, expected_method, expected_apparent_source, record_route):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, DynamicWorkflowConfig(rw=0.05), record_route, focus_steps=["porosity"], curves=["PHIN", "PHID", "PHIS", "PHIT", "PHIE"])
    step = _step_by_name(result, "porosity")
    assert step.parameters["phit_method"] == expected_method
    assert "PHIE" in step.output_curves
    assert step.parameters["phit_apparent"] is bool(expected_apparent_source)
    assert step.parameters["phie_apparent"] is bool(expected_apparent_source)
    assert step.parameters["apparent_porosity_source"] == expected_apparent_source

    phit = well.get("PHIT")
    phie = well.get("PHIE")
    assert phit is not None
    assert phie is not None
    assert phit.metadata.custom["porosity_apparent"] is bool(expected_apparent_source)
    assert phie.metadata.custom["porosity_apparent"] is bool(expected_apparent_source)
    if expected_apparent_source:
        assert expected_apparent_source in phit.metadata.description
        assert expected_apparent_source in phie.metadata.description
        assert any("apparent total porosity" in item for item in step.decisions)
        assert any("apparent effective porosity" in item for item in step.decisions)
    else:
        assert "Apparent" not in phit.metadata.description


def test_porosity_phin_uses_lithology_specific_nphi_ma(record_route):
    well = _routing_well("porosity_lithology_specific_phin", n=4)
    well.add_curve("LITHOLOGY_GEOLINK", np.array([1, 7, 11, 15]), "", "Lithology")

    result = _run(
        "porosity_lithology_specific_phin",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["porosity"],
        curves=["NPHI", "PHIN"],
    )

    step = _step_by_name(result, "porosity")
    expected_nphi_ma = np.array([-0.02, 0.30, 0.00, 0.02])
    expected_phin = np.clip(well.get("NPHI").data - expected_nphi_ma, 0.0, 0.6)
    np.testing.assert_allclose(well.get("PHIN").data, expected_phin)
    assert step.parameters["matrix_mode"] == "lithology_dependent"
    endpoints = {
        item["lithology_group"]: item["nphi_ma"]
        for item in step.parameters["matrix_endpoints"]
    }
    assert endpoints["SANDSTONE"] == -0.02
    assert endpoints["SHALE"] == 0.30
    assert endpoints["LIMESTONE"] == 0.00
    assert endpoints["DOLOMITE"] == 0.02
    assert any("lithology-specific matrix neutron endpoints" in item for item in step.decisions)


def test_canonical_workflow_keeps_single_porosity_pass(record_route):
    well = _routing_well("porosity_refined_from_computed_lithology", n=8)

    result = _run(
        "porosity_refined_from_computed_lithology",
        well,
        DynamicWorkflowConfig(
            rw=0.05,
            matrix_type="sandstone",
            skip_steps=[
                "temperature", "bad_hole", "vclay", "rcl", "rmf", "hc_correction",
                "core_correction", "rw", "invasion", "saturation", "permeability",
                "rock_quality", "bvw", "mhi", "net_pay",
            ],
        ),
        record_route,
        focus_steps=["porosity", "lithology", "porosity_refinement"],
        curves=["LITHO", "PHIN", "PHID", "PHIT", "PHIE"],
    )

    initial = _step_by_name(result, "porosity")

    assert initial.parameters["porosity_pass"] == "initial"
    assert initial.parameters["matrix_mode"] == "constant"
    assert _step_or_none(result, "porosity_refinement") is None
    assert "porosity_refinement" in result.steps_skipped
    assert well.get("LITHO") is not None


def test_porosity_refinement_skips_when_porosity_step_is_skipped():
    well = _routing_well("porosity_refinement_skip_follows_porosity")

    result = DynamicWorkflow(
        DynamicWorkflowConfig(
            rw=0.05,
            skip_steps=[
                "temperature", "bad_hole", "vclay", "rcl", "rmf", "hc_correction",
                "porosity", "core_correction", "rw", "invasion", "saturation",
                "permeability", "rock_quality", "bvw", "mhi", "net_pay",
            ],
        )
    ).run(well)

    assert "porosity" in result.steps_skipped
    assert "porosity_refinement" in result.steps_skipped
    assert _step_or_none(result, "porosity_refinement") is None


def test_porosity_phit_falls_back_to_phid_where_nphi_missing(record_route):
    well = _routing_well("porosity_partial_nphi")
    nphi = well.get("NPHI").data
    nphi[::2] = np.nan

    result = _run(
        "porosity_partial_nphi",
        well,
        DynamicWorkflowConfig(rw=0.05),
        record_route,
        focus_steps=["porosity"],
        curves=["NPHI", "PHIN", "PHID", "PHIT", "PHIE"],
    )

    step = _step_by_name(result, "porosity")
    counts = step.parameters["phit_source_counts"]
    missing_phin_mask = ~np.isfinite(well.get("PHIN").data) & np.isfinite(well.get("PHID").data)

    assert step.parameters["phit_method"] == "adaptive_nd"
    assert counts["density_from_missing_phin"] == int(np.count_nonzero(missing_phin_mask))
    assert counts["density_from_missing_phin"] > 0
    np.testing.assert_allclose(
        well.get("PHIT").data[missing_phin_mask],
        well.get("PHID").data[missing_phin_mask],
    )


@pytest.mark.parametrize(
    "case_id,config,expected_model,expected_counts",
    [
        (
            "saturation_auto_mixed_archie_simandoux",
            DynamicWorkflowConfig(
                rw=0.05,
                rmf_at_meas_temp=0.2,
                salinity_ppm=35000.0,
                sw_model="auto",
                vcl_cutoff=0.3,  # pin the routing threshold; the standard default is 0.40
                lithology="clastic",
                skip_steps=["vclay", "lithology"],
            ),
            "archie+simandoux",
            {"archie": 4, "simandoux": 8},
        ),
        (
            "saturation_forced_archie",
            DynamicWorkflowConfig(
                rw=0.05,
                sw_model="archie",
                lithology="clastic",
                skip_steps=["vclay", "lithology"],
            ),
            "archie",
            {"archie": 12},
        ),
    ],
)
def test_saturation_routes_by_model_and_vcl(case_id, config, expected_model, expected_counts, record_route):
    well = _routing_well(case_id)
    result = _run(case_id, well, config, record_route, focus_steps=["saturation"], curves=["SW", "SXO", "RW", "RMF"])
    step = _step_by_name(result, "saturation")
    assert step.method == expected_model
    assert step.parameters["sw_model_sample_counts"] == expected_counts
    assert well.get("SW") is not None


def test_saturation_uses_vclay_when_vcl_alias_is_missing(record_route):
    well = _remove_curves(_routing_well("SAT-VCLAY-ONLY"), "VCL")
    result = _run(
        "saturation_vclay_only",
        well,
        DynamicWorkflowConfig(
            rw=0.05,
            salinity_ppm=35000.0,
            sw_model="auto",
            vcl_cutoff=0.3,  # pin the routing threshold; the standard default is 0.40
            skip_steps=["vclay", "lithology"],
        ),
        record_route,
        focus_steps=["saturation"],
        curves=["SW", "SXO", "VCLAY"],
    )
    step = _step_by_name(result, "saturation")

    assert step.parameters["sw_model_sample_counts"] == {"archie": 4, "simandoux": 8}
    assert step.input_curves == ["RDEP", "PHIE", "VCLAY"]


def test_saturation_can_write_sxo_without_sw_inputs(record_route):
    well = _remove_curves(_routing_well("SAT-SXO-NO-SW"), "RDEP", "RT", "SW", "SXO")
    result = _run(
        "saturation_sxo_without_sw_inputs",
        well,
        DynamicWorkflowConfig(
            rmf_at_meas_temp=0.08,
            rmf_meas_temp_f=75.0,
            sw_model="archie",
            skip_steps=[
                "vclay",
                "rcl",
                "lithology",
                "rw",
                "invasion",
                "hc_correction",
                "porosity",
                "permeability",
                "rock_quality",
                "bvw",
                "mhi",
                "net_pay",
            ],
        ),
        record_route,
        focus_steps=["rmf", "saturation"],
        curves=["RMF", "SW", "SXO"],
    )
    step = _step_by_name(result, "saturation")

    assert step.method == "not_computed_missing_rt"
    assert step.parameters["sxo_method"] == "archie_rxo_rmf"
    assert step.parameters["sw_model_sample_counts"] == {}
    assert np.isnan(well.get("SW").data).all()
    assert np.isfinite(well.get("SXO").data).all()


def test_dual_water_uses_same_phi_cl_as_phie(record_route):
    well = _routing_well("SAT-PHICL-PHIT")
    result = _run(
        "saturation_dual_water_phi_cl_from_phit",
        well,
        DynamicWorkflowConfig(
            rw=0.05,
            rwb=0.02,  # Dual Water runs only with a supplied Rwb
            sw_model="auto",
            lithology="carbonate",
            skip_steps=["vclay", "lithology"],
        ),
        record_route,
        focus_steps=["porosity", "saturation"],
        curves=["PHIT", "PHIE", "SW"],
    )

    por_step = _step_by_name(result, "porosity")
    sat_step = _step_by_name(result, "saturation")
    vcl = well.get("VCL").data
    phit = well.get("PHIT").data
    expected_phi_cl = float(np.nanmean(phit[np.isfinite(vcl) & (vcl > 0.9)]))

    assert por_step.parameters["phi_cl_source"] == "vcl_gt_0.9_average_phit_nd"
    assert sat_step.parameters["phi_cl_source"] == "vcl_gt_0.9_average_phit_nd"
    assert np.isscalar(por_step.parameters["phi_cl"])
    assert por_step.parameters["phi_cl"] == pytest.approx(expected_phi_cl)
    assert sat_step.parameters["phi_cl"] == pytest.approx(por_step.parameters["phi_cl"])
    assert sat_step.parameters["sw_model_sample_counts"] == {"archie": 4, "dual_water": 8}


def test_phi_cl_is_one_representative_scalar_across_depth():
    n = 601
    well = Well("PHICL-SCALAR")
    well.depth = np.linspace(1000.0, 2000.0, n)
    well.add_curve("DEPT", well.depth, "m", "Depth")
    well.add_curve("VCL", np.full(n, 0.95), "v/v", "Clay volume")
    phit = np.linspace(0.30, 0.10, n)

    workflow = DynamicWorkflow(DynamicWorkflowConfig(phi_cl=0.15))
    info = workflow._resolve_phi_cl(
        well,
        well.get("VCL"),
        porosity_basis=phit,
        porosity_basis_name="PHIT",
        porosity_basis_valid_mask=np.ones(n, dtype=bool),
    )

    assert np.isscalar(info["phi_cl"])
    assert info["phi_cl"] == pytest.approx(float(np.mean(phit)))
    assert info["phi_cl_source"] == "vcl_gt_0.9_average_phit_nd"
    assert "phi_cl_values" not in info


def test_phi_cl_uses_config_fallback_when_shale_phit_is_not_nd_average(record_route):
    well = _routing_well("PHICL-FALLBACK-NO-ND-SHALE")
    well.get("NPHI").data[np.asarray(well.get("VCL").data) > 0.9] = np.nan

    result = _run(
        "phi_cl_fallback_when_shale_phit_not_nd_average",
        well,
        DynamicWorkflowConfig(rw=0.05, phi_cl=0.15, skip_steps=["vclay", "lithology"]),
        record_route,
        focus_steps=["porosity", "saturation"],
        curves=["NPHI", "PHIN", "PHID", "PHIT", "PHIE", "SW"],
    )

    por_step = _step_by_name(result, "porosity")
    sat_step = _step_by_name(result, "saturation")

    assert por_step.parameters["phit_source_counts"]["density_from_missing_phin"] > 0
    assert por_step.parameters["phi_cl_source"] == "config_or_default"
    assert por_step.parameters["phi_cl"] == pytest.approx(0.15)
    assert "no shale-control PHIT samples" in por_step.parameters["phi_cl_fallback_reason"]
    assert sat_step.parameters["phi_cl_source"] == "config_or_default"
    assert sat_step.parameters["phi_cl"] == pytest.approx(por_step.parameters["phi_cl"])


def test_saturation_skips_when_phie_is_unavailable_even_if_rhob_exists(record_route):
    well = _remove_curves(_routing_well("SAT-MISSING-PHIE"), "PHIE", "PHIT")
    result = _run(
        "saturation_missing_phie",
        well,
        DynamicWorkflowConfig(rw=0.05, skip_steps=["porosity"]),
        record_route,
        focus_steps=["saturation"],
        curves=["SW", "PHIE", "RHOB"],
    )

    assert "saturation" in result.steps_skipped
    assert _step_or_none(result, "saturation") is None


@pytest.mark.parametrize(
    "case_id,mutate,config,expected_swirr_source",
    [
        ("permeability_user_swirr", lambda well: well, DynamicWorkflowConfig(perm_method="timur", swirr=0.25), "user"),
        ("permeability_swirr_curve", lambda well: (well.add_curve("SWIRR", np.full(well.n_samples, 0.22), "v/v", "Swirr"), well)[-1], DynamicWorkflowConfig(perm_method="timur"), "swirr"),
        (
            "permeability_nmr_swirr",
            lambda well: (
                _remove_curves(well, "SWIRR", "SWI", "SWIR"),
                well.add_curve("TCMR", np.full(well.n_samples, 0.30), "v/v", "Total CMR Porosity"),
                well.add_curve("CMFF", np.full(well.n_samples, 0.21), "v/v", "CMR Free Fluid"),
                well,
            )[-1],
            DynamicWorkflowConfig(perm_method="timur", skip_steps=["vclay", "porosity", "saturation"]),
            "nmr_derived",
        ),
        (
            # The benchmark profile counts the two washed-out samples, which
            # take the hydrocarbon mask past its 10-sample minimum.
            "permeability_hc_mask_p5",
            lambda well: (
                _remove_curves(well, "SWIRR", "SWI", "SWIR"),
                well.add_curve("VCL", np.full(well.n_samples, 0.10), "v/v", "Clay Volume"),
                well.add_curve("PHIE", np.full(well.n_samples, 0.18), "v/v", "Effective Porosity"),
                well.add_curve("SW", np.linspace(0.18, 0.40, well.n_samples), "v/v", "Water Saturation"),
                well.add_curve("RDEP", np.full(well.n_samples, 12.0), "ohm.m", "Deep Resistivity"),
                well,
            )[-1],
            DynamicWorkflowConfig(
                perm_method="timur", skip_steps=["vclay", "porosity", "saturation"],
                calculation_profile="benchmark",
            ),
            "hc_mask_p5",
        ),
        (
            # The current profile leaves bad-hole samples out of the mask; in
            # gauge hole every sample counts.
            "permeability_hc_mask_p5_gauge_hole",
            lambda well: (
                _remove_curves(well, "SWIRR", "SWI", "SWIR", "CALI", "CAL", "DRHO"),
                well.add_curve("VCL", np.full(well.n_samples, 0.10), "v/v", "Clay Volume"),
                well.add_curve("PHIE", np.full(well.n_samples, 0.18), "v/v", "Effective Porosity"),
                well.add_curve("SW", np.linspace(0.18, 0.40, well.n_samples), "v/v", "Water Saturation"),
                well.add_curve("RDEP", np.full(well.n_samples, 12.0), "ohm.m", "Deep Resistivity"),
                well,
            )[-1],
            DynamicWorkflowConfig(perm_method="timur", skip_steps=["vclay", "porosity", "saturation"]),
            "hc_mask_p5",
        ),
        ("permeability_default_swirr_020", lambda well: (well.add_curve("RDEP", np.full(well.n_samples, 4.0), "ohm.m", "Deep Resistivity"), _remove_curves(well, "SWIRR", "SWI", "SWIR"))[-1], DynamicWorkflowConfig(perm_method="timur", skip_steps=["vclay", "porosity", "saturation"]), "default_0.20"),
    ],
)
def test_permeability_routes_swirr_sources(case_id, mutate, config, expected_swirr_source, record_route):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, config, record_route, focus_steps=["permeability"], curves=["PERM", "RESFLAG", "SWIRR_APP"])
    step = _step_by_name(result, "permeability")
    assert step.parameters["swirr_source"] == expected_swirr_source
    if expected_swirr_source in {"hc_mask_p5", "default_0.20"}:
        assert step.parameters["swirr_metadata"]["resflag_source"] == "RESFLAG"
        assert step.parameters["pre_permeability_resflag"]["created"] is True
        assert well.get("RESFLAG") is not None
        swirr_app = well.get("SWIRR_APP")
        assert swirr_app is not None
        assert "SWIRR_APP" in step.output_curves
        assert step.parameters["swirr_app_method"] == expected_swirr_source
        np.testing.assert_allclose(swirr_app.data, step.parameters["swirr"])
    if expected_swirr_source == "default_0.20":
        assert step.parameters["swirr"] == 0.20
    if expected_swirr_source == "nmr_derived":
        assert well.get("SWIRR_NMR") is None
        assert well.get("SWIRR_APP") is None
        assert "SWIRR_NMR" not in step.output_curves
        assert "RESFLAG" not in step.output_curves
        assert step.parameters["swirr_metadata"]["method"] == "nmr_total_minus_ffi_over_total_porosity"
        assert {"TCMR", "CMFF"} <= set(step.input_curves)
    assert well.get("PERM") is not None


def test_permeability_routes_lith_vcl_empirical_method(record_route):
    well = _routing_well("PERM-LITH-VCL")
    well.add_curve(
        "LITHOLOGY_GEOLINK",
        np.array([11, 12, 13, 14, 15, 17, 24, 20, 16, 1, 1, 1]),
        "",
        "Lithology Code",
    )
    result = _run(
        "permeability_lith_vcl",
        well,
        DynamicWorkflowConfig(
            perm_method="lith_vcl",
            skip_steps=["vclay", "porosity", "saturation"],
        ),
        record_route,
        focus_steps=["permeability"],
        curves=["PERM", "RESFLAG"],
    )
    step = _step_by_name(result, "permeability")

    assert step.parameters["empirical_method"] == "lith_vcl"
    assert step.parameters["swirr_source"] == "not_required"
    assert "RESFLAG" not in step.output_curves
    assert "PHIT" in step.input_curves
    assert "VCL" in step.input_curves
    assert "LITHOLOGY_GEOLINK" in step.input_curves
    assert well.get("PERM") is not None


def test_rock_quality_bvw_mhi_and_net_pay_use_canonical_curves(record_route):
    well = _routing_well("RESERVOIR-INDEX")
    result = _run(
        "reservoir_index_modules_canonical_curves",
        well,
        DynamicWorkflowConfig(rw=0.05, rmf_at_meas_temp=0.2, perm_method="timur", swirr=0.25),
        record_route,
        focus_steps=["rock_quality", "bvw", "mhi", "net_pay"],
        curves=["RQI", "FZI", "PHI_Z", "BVW", "MHI", "NET_ROCK", "NET_RES", "NET_PAY", "SEALFLAG"],
    )

    assert _step_by_name(result, "rock_quality").method == "RQI_FZI_phi_z"
    assert _step_by_name(result, "bvw").method == "PHIE_x_SW"
    assert _step_by_name(result, "mhi").method == "moveable_HC_index"
    assert _step_by_name(result, "net_pay").method == "cutoff_flags"
    assert well.get("BVW") is not None
    assert well.get("MHI") is not None
    assert well.get("NET_PAY") is not None


def test_mhi_skips_when_sxo_is_only_fifth_root_fallback(record_route):
    well = _remove_curves(_routing_well("MHI-FALLBACK-SXO"), "RSHA", "RXO", "RESS", "RMF", "SXO")

    result = _run(
        "mhi_fallback_sxo_skipped",
        well,
        DynamicWorkflowConfig(
            rw=0.05,
            sw_model="archie",
            skip_steps=["vclay", "rcl", "lithology", "rmf", "invasion", "hc_correction", "porosity", "permeability", "rock_quality", "bvw", "net_pay"],
        ),
        record_route,
        focus_steps=["saturation", "mhi"],
        curves=["SW", "SXO", "MHI"],
    )

    saturation_step = _step_by_name(result, "saturation")
    mhi_step = _step_by_name(result, "mhi")

    assert saturation_step.parameters["sxo_method"] == "fifth_root_quicklook"
    assert well.get("SXO") is not None
    assert well.get("MHI") is None
    assert mhi_step.output_curves == []
    assert mhi_step.parameters["diagnostic_sxo_samples"] == 0
    assert "not computed" in mhi_step.decisions[0]


@pytest.mark.parametrize(
    "case_id,path,expected_methods,expected_skips,expected_curves",
    [
        (
            "bundled_geolink_15_9_14",
            sample("15_9-14"),
            {
                "temperature": "fallback_geothermal_gradient",
                "rcl": "lithology_shale_median_rt",
                "rw": "rwa_clean_wet_p5",
            },
            {"rmf", "hc_correction"},
            {"TEMP", "VCL", "RW", "PHIE", "SW", "SXO", "PERM", "BVW", "NET_PAY"},
        ),
        (
            "bundled_geolink_16_7_1",
            sample("16_7-1"),
            {
                "temperature": "fallback_geothermal_gradient",
                "rcl": "lithology_shale_median_rt",
                "rw": "rwa_clean_wet_p5",
            },
            {"rmf", "hc_correction"},
            {"TEMP", "VCL", "RW", "PHIE", "SW", "SXO", "PERM", "BVW", "NET_PAY"},
        ),
    ],
)
def test_real_las_wells_route_through_dynamic_workflow(
    case_id, path, expected_methods, expected_skips, expected_curves, record_route
):
    well = read_las(path)
    result = _run(
        case_id,
        well,
        DynamicWorkflowConfig(),
        record_route,
        focus_steps=[
            "temperature", "bad_hole", "vclay", "rcl", "lithology", "rmf", "rw",
            "invasion", "hc_correction", "porosity", "saturation", "permeability",
            "rock_quality", "bvw", "mhi", "net_pay",
        ],
        curves=sorted(expected_curves | {"RMF", "INVASION_RATIO"}),
        notes=f"source_las={path}",
    )

    assert result.success
    assert expected_skips.issubset(set(result.steps_skipped))
    for step_name, method in expected_methods.items():
        assert _step_by_name(result, step_name).method == method
    for curve_name in expected_curves:
        assert well.get(curve_name) is not None, f"{case_id} missing {curve_name}"


def test_reused_workflow_resolves_auto_matrix_per_well():
    import petrophys as pp

    def config():
        return DynamicWorkflowConfig(matrix_type="auto", lithology="auto")

    fresh = DynamicWorkflow(config()).run(pp.read(pp.sample("15_9-14")))

    workflow = DynamicWorkflow(config())
    carbonate = workflow.run(pp.read(pp.sample("HW-3")))
    reused = workflow.run(pp.read(pp.sample("15_9-14")))

    assert carbonate.config.matrix_type != fresh.config.matrix_type
    assert (reused.config.matrix_type, reused.config.lithology) == (
        fresh.config.matrix_type,
        fresh.config.lithology,
    )
    assert (workflow.config.matrix_type, workflow.config.lithology) == ("auto", "auto")


def test_skipped_steps_explain_what_is_missing(capsys):
    import petrophys as pp

    result = pp.interpret(pp.read(pp.sample("HW-3")), rw=0.07)
    reasons = result.specific_steps_skipped

    assert reasons["hc_correction"] == "apply_hc_correction=False (hydrocarbon correction is opt-in)"
    assert reasons["core_correction"].startswith("Missing core porosity")
    assert "mud_type='OBM'" in reasons["rmf"]
    assert all("Required usable curves" not in reason for reason in reasons.values())

    disabled = pp.interpret(pp.read(pp.sample("HW-3")), rw=0.07, skip_steps=["bvw"])
    assert disabled.specific_steps_skipped["bvw"] == "Disabled in skip_steps"

    result.print_trace()
    assert "hc_correction: apply_hc_correction=False" in capsys.readouterr().out


def test_explicit_carbonate_domain_keeps_a_carbonate_matrix():
    import petrophys as pp

    well = pp.read(pp.sample("HW-27"))  # core grain density alone points to sandstone
    result = pp.interpret(well, lithology_domain="carbonate")
    assert result.config.matrix_type in {"limestone", "dolomite"}
    assert any("lithology_domain is 'carbonate', but" in warning for warning in result.warnings)


def test_default_domain_is_clastic_and_auto_domain_follows_the_evidence():
    import petrophys as pp

    well = pp.read(pp.sample("HW-3"))  # core grain density says dolomite
    default = pp.interpret(well)
    assert (default.config.lithology_domain, default.config.matrix_type) == ("clastic", "sandstone")

    auto = pp.interpret(well, lithology_domain="auto")
    assert (auto.config.lithology_domain, auto.config.matrix_type) == ("carbonate", "dolomite")


def test_net_pay_decisions_use_the_well_depth_unit():
    import petrophys as pp

    feet = pp.interpret(pp.read(pp.sample("HW-3")), lithology_domain="carbonate")
    metres = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
    for result, unit in ((feet, "ft"), (metres, "m")):
        decisions = [d for s in result.steps_executed if s.step == "net_pay" for d in s.decisions
                     if d.startswith("NET_PAY")]
        assert decisions and decisions[0].endswith(f" {unit})"), decisions
