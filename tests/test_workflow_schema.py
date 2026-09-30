"""Machine-readable workflow descriptions stay complete and serializable."""

import dataclasses
import json

from petrophys.workflow import DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import WorkflowStep
from petrophys.workflow.schema import config_schema, describe_setting, workflow_steps


def test_every_setting_is_described():
    schema = config_schema()
    assert [item["name"] for item in schema] == [
        f.name for f in dataclasses.fields(DynamicWorkflowConfig)
    ]
    assert all(item["description"] for item in schema)
    json.dumps(schema)


def test_choices_match_validation():
    for item in config_schema():
        for choice in item.get("choices", ()):
            if item["name"] == "skip_steps":
                DynamicWorkflowConfig(skip_steps=[choice])
            else:
                DynamicWorkflowConfig(**{item["name"]: choice})


def test_workflow_steps_are_ordered_and_described():
    steps = workflow_steps()
    assert {step["name"] for step in steps} == {step.value for step in WorkflowStep}
    assert steps[0]["name"] == "normalization" and steps[-1]["name"] == "net_pay"
    assert all(step["description"] for step in steps)
    assert describe_setting("bht_f")["type"] == "Optional[float]"


def test_well_and_result_summaries_are_json_ready():
    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"))
    described = well.describe()
    json.dumps(described)
    assert described["depth_unit"] == "m"
    assert {"mnemonic", "unit", "description", "valid_fraction"} <= set(described["curves"][0])

    result = pp.interpret(well, rw=0.07)
    summary = result.summary(top=2841, bottom=2958)
    json.dumps(summary)
    assert summary["status"] == "interpreted"
    assert summary["settings_used"]["rw"] == 0.07
    assert summary["net_thickness"]["net_pay"] > 0
    assert 0.1 < summary["key_results"]["PHIE"]["median"] < 0.3
    assert summary["skipped_steps"]["hc_correction"].startswith("apply_hc_correction=False")


def test_formation_summary_uses_attached_or_given_formations():
    import pytest

    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    formations = result.formation_summary(rank="formation")
    assert "Etive Fm." in set(formations["Formation"])
    assert not any(name.endswith("GP.") for name in formations["Formation"])
    assert (formations["Net Pay (m)"] >= 0).all()

    bare = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
    with pytest.raises(ValueError, match="No formations are attached"):
        bare.formation_summary()
    assert len(bare.formation_summary({"UPPER": 2000.0, "BRENT": 2841.0})) == 2


def test_skip_reasons_point_to_the_right_tool():
    import petrophys as pp

    result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
    reasons = result.specific_steps_skipped
    assert "interpret_spectral_gamma" in reasons["spectral_gamma"]
    assert all("canonical" not in reason for reason in reasons.values())
