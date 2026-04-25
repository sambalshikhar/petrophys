import json

import numpy as np

from petrophys import Well
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, StepResult


def _trace_well():
    depth = np.arange(1000.0, 1010.0, dtype=float)
    well = Well("TRACE-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.header.td = float(depth[-1])

    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.linspace(40.0, 120.0, depth.size), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.18, 2.45, depth.size), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.28, 0.12, depth.size), "v/v", "Neutron Porosity")
    well.add_curve("RDEP", np.linspace(60.0, 8.0, depth.size), "ohm.m", "Deep Resistivity")
    well.add_curve("PHIE", np.linspace(0.24, 0.12, depth.size), "v/v", "Effective Porosity")
    well.add_curve("PHIT", np.linspace(0.26, 0.14, depth.size), "v/v", "Total Porosity")
    well.add_curve("VCL", np.linspace(0.05, 0.45, depth.size), "v/v", "Clay Volume")
    return well


def test_step_result_to_dict_is_json_safe():
    step = StepResult(
        step="demo",
        success=True,
        method="array_summary",
        input_curves=["A"],
        output_curves=["B"],
        parameters={"scalar": np.float64(1.5), "arr": np.array([1.0, np.nan, 3.0])},
        decisions=["used test route"],
    )

    payload = step.to_dict()

    assert payload["parameters"]["scalar"] == 1.5
    assert payload["parameters"]["arr"]["finite_count"] == 2
    assert payload["parameters"]["arr"]["nan_count"] == 1
    json.dumps(payload, allow_nan=False)


def test_dynamic_workflow_result_exports_trace_dict_json_and_markdown(tmp_path):
    well = _trace_well()
    config = DynamicWorkflowConfig(
        bht_f=165.0,
        td_ft=float(well.depth[-1]),
        rw=0.05,
        rw_meas_temp_f=75.0,
        perm_method="timur",
        swirr=0.25,
        lithology="clastic",
    )

    result = DynamicWorkflow(config).run(well)
    payload = result.to_trace_dict(include_curves=["TEMP", "RW", "SW"])

    assert payload["schema"] == "dynamic_workflow_trace"
    assert payload["well"]["name"] == "TRACE-1"
    assert "temperature" in payload["executed_steps"]
    assert "rw" in payload["executed_steps"]
    assert payload["steps"]
    assert all(
        {"step", "success", "method", "input_curves", "output_curves", "parameters", "decisions"}
        <= set(step)
        for step in payload["steps"]
    )
    assert "TEMP" in payload["curve_summaries"]
    assert payload["curve_summaries"]["TEMP"]["data"]["finite_count"] == well.n_samples

    json_text = result.to_json(include_curves=["TEMP", "RW"])
    parsed = json.loads(json_text)
    assert parsed["well"]["name"] == "TRACE-1"

    json_path = result.to_json(tmp_path / "trace.json", include_curves=["TEMP", "RW"])
    assert json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1

    markdown_path = result.to_markdown(tmp_path / "trace.md", include_curves=["TEMP"])
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Dynamic Workflow Trace" in markdown
    assert "## temperature" in markdown
    assert "## Curve Summaries" in markdown
