import json
import os
from pathlib import Path

import numpy as np
import pytest

from petrophys import Well
from petrophys.io import read_las
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig


REPORT_DIR = Path(os.environ.get("WORKFLOW_ROUTING_LOG_DIR", "reports/workflow_routing_tests"))
JSONL_REPORT = REPORT_DIR / "routing_cases.jsonl"
MARKDOWN_REPORT = REPORT_DIR / "routing_summary.md"


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


@pytest.fixture(scope="session", autouse=True)
def _reset_workflow_routing_report():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_REPORT.unlink(missing_ok=True)
    MARKDOWN_REPORT.write_text(
        "# Dynamic Workflow Routing Test Log\n\n"
        "Each section records the workflow route, method, parameters, decisions, "
        "and output curve summaries for one pytest case.\n",
        encoding="utf-8",
    )
    yield
    print(f"\nDynamic workflow routing report: {MARKDOWN_REPORT}")


@pytest.fixture
def record_route():
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
        with JSONL_REPORT.open("a", encoding="utf-8") as handle:
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
        MARKDOWN_REPORT.write_text(
            MARKDOWN_REPORT.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
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
        "invasion", "hc_correction", "porosity", "saturation", "permeability",
        "rock_quality", "bvw", "mhi", "net_pay",
    }
    executed = {step.step for step in result.steps_executed if step.success}
    assert expected_steps.issubset(executed)
    assert result.success


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

    assert _step_by_name(result, "temperature").method == "fallback_geothermal_gradient"
    assert _step_by_name(result, "rw").method == "default_fallback_profile"
    assert {
        "bad_hole", "vclay", "rcl", "lithology", "rmf", "invasion",
        "hc_correction", "porosity", "saturation", "permeability",
        "rock_quality", "bvw", "mhi", "net_pay",
    }.issubset(set(result.steps_skipped))


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
        ("vclay_gr_sp_nd_enabled", lambda well: well, DynamicWorkflowConfig(), True, True),
        ("vclay_sp_disabled_by_kcl_mud", lambda well: well, DynamicWorkflowConfig(mud_has_clay_or_kcl=True), False, True),
        ("vclay_nd_disabled_when_missing_density_neutron", lambda well: _remove_curves(well, "NPHI", "RHOB"), DynamicWorkflowConfig(), True, False),
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
        curves=["BAD_HOLE", "BADHOLEFLAG"],
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
        curves=["BAD_HOLE", "BADHOLEFLAG"],
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
            "rw_las_rw_temp_c_corrected",
            lambda well: (well.set_parameter("RW", 0.05), well.set_parameter("RWT", 25.0), well.set_parameter("RWT__UNIT", "C"), well)[-1],
            DynamicWorkflowConfig(),
            "las_metadata_temperature_profile_corrected",
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
    if case_id == "rw_rwa_clean_wet_fallback":
        assert "rwa_reference_temp_f" in step.parameters
        assert "salinity_ppm_estimated" in step.parameters
        assert step.parameters["rw_min"] < step.parameters["rw_max"]
        rdep = well.get("RDEP").data
        phie = well.get("PHIE").data
        vcl = well.get("VCL").data
        valid = np.isfinite(rdep) & np.isfinite(phie) & np.isfinite(vcl) & (vcl < 0.3) & (phie > 0.05)
        expected_rwa = rdep[valid] * np.power(phie[valid], step.parameters["archie_m"]) / step.parameters["archie_a"]
        assert np.isclose(step.parameters["rw_from_rwa"], np.percentile(expected_rwa, 5))


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
    "case_id,mutate,expected_method",
    [
        ("porosity_full_density_neutron_sonic", lambda well: well, "adaptive_nd"),
        ("porosity_density_only", lambda well: _remove_curves(well, "NPHI", "DTC", "PHIN", "PHIS", "PHIT"), "density_only"),
        ("porosity_existing_phit_reused", lambda well: _remove_curves(well, "NPHI", "RHOB", "DTC", "PHIN", "PHID", "PHIS"), "existing_phit"),
    ],
)
def test_porosity_routes_by_available_inputs(case_id, mutate, expected_method, record_route):
    well = mutate(_routing_well(case_id))
    result = _run(case_id, well, DynamicWorkflowConfig(rw=0.05), record_route, focus_steps=["porosity"], curves=["PHIN", "PHID", "PHIS", "PHIT", "PHIE"])
    step = _step_by_name(result, "porosity")
    assert step.parameters["phit_method"] == expected_method
    assert "PHIE" in step.output_curves


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
            skip_steps=["vclay", "lithology"],
        ),
        record_route,
        focus_steps=["saturation"],
        curves=["SW", "SXO", "VCLAY"],
    )
    step = _step_by_name(result, "saturation")

    assert step.parameters["sw_model_sample_counts"] == {"archie": 4, "simandoux": 8}
    assert step.input_curves == ["RDEP", "PHIE", "VCLAY"]


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
            "permeability_hc_mask_p5",
            lambda well: (
                _remove_curves(well, "SWIRR", "SWI", "SWIR"),
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
        DynamicWorkflowConfig(rw=0.05, perm_method="timur", swirr=0.25),
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


@pytest.mark.parametrize(
    "case_id,path,expected_methods,expected_skips,expected_curves",
    [
        (
            "real_geolink_15_9_12",
            Path("geolink_data/15_9-12.las"),
            {
                "temperature": "fallback_geothermal_gradient",
                "rcl": "lithology_shale_median_rt",
                "rw": "rwa_clean_wet_p5",
            },
            {"rmf", "hc_correction"},
            {"TEMP", "VCL", "RW", "PHIE", "SW", "SXO", "PERM", "BVW", "MHI", "NET_PAY"},
        ),
        (
            "real_geolink_15_9_23",
            Path("geolink_data/15_9-23.las"),
            {
                "temperature": "fallback_geothermal_gradient",
                "rcl": "lithology_shale_median_rt",
                "rw": "rwa_clean_wet_p5",
            },
            {"rmf", "invasion", "hc_correction"},
            {"TEMP", "VCL", "RW", "PHIE", "SW", "SXO", "PERM", "BVW", "MHI", "NET_PAY"},
        ),
        (
            "real_eval_35_8_2",
            Path("C:/Users/HP/Desktop/Plots/LAS_with_petrophys_params/eval/35_8-2.las"),
            {
                "temperature": "linear_gradient_from_BHT",
                "rmf": "las_metadata_profile_temperature_correction",
                "rw": "default_fallback_profile",
            },
            {
                "bad_hole", "vclay", "rcl", "lithology", "invasion", "hc_correction",
                "porosity", "saturation", "permeability", "rock_quality", "bvw", "mhi", "net_pay",
            },
            {"TEMP", "RW", "RMF"},
        ),
        (
            "real_eval_25_10_9",
            Path("C:/Users/HP/Desktop/Plots/LAS_with_petrophys_params/eval/25_10-9.las"),
            {
                "temperature": "linear_gradient_from_BHT",
                "rcl": "lithology_shale_median_rt",
                "rw": "rwa_clean_wet_p5",
            },
            {"rmf", "invasion", "hc_correction"},
            {"TEMP", "VCL", "RW", "PHIE", "SW", "SXO", "PERM", "BVW", "MHI", "NET_PAY"},
        ),
    ],
)
def test_real_las_wells_route_through_dynamic_workflow(
    case_id, path, expected_methods, expected_skips, expected_curves, record_route
):
    if not path.exists():
        pytest.skip(f"Real LAS test data not available: {path}")

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
