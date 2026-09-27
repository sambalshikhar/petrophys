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
