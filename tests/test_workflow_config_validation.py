"""DynamicWorkflowConfig rejects invalid settings instead of guessing."""

import dataclasses

import pytest

from petrophys.workflow import DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import WorkflowStep


def test_defaults_resolve_matrix_and_lithology_from_data():
    config = DynamicWorkflowConfig()
    assert (config.matrix_type, config.lithology) == ("auto", "auto")
    assert config.skip_steps == []


@pytest.mark.parametrize(
    ("field", "value", "suggestion"),
    [
        ("hc_type", "gass", "gas"),
        ("matrix_type", "sandstne", "sandstone"),
        ("lithology", "carbonite", "carbonate"),
        ("sw_model", "archi", "archie"),
        ("mud_water", "salty", "salt"),
        ("gr_method", "larionov_tertary", "larionov_tertiary"),
        ("perm_method", "timurr", "timur"),
    ],
)
def test_misspelled_choices_are_rejected_with_a_suggestion(field, value, suggestion):
    with pytest.raises(ValueError, match=f"did you mean '{suggestion}'"):
        DynamicWorkflowConfig(**{field: value})


def test_unknown_skip_step_is_rejected():
    with pytest.raises(ValueError, match="did you mean 'permeability'"):
        DynamicWorkflowConfig(skip_steps=["permeabilty"])


def test_skip_steps_must_be_a_list():
    with pytest.raises(ValueError, match="skip_steps must be a list"):
        DynamicWorkflowConfig(skip_steps="bvw")


def test_choices_are_normalized():
    config = DynamicWorkflowConfig(
        mud_type="obm",
        hc_type="Gas",
        sw_model="ARCHIE",
        lithology="Carbonate",
        skip_steps=[WorkflowStep.BVW, "NET_PAY"],
    )
    assert config.mud_type == "OBM"
    assert config.hc_type == "gas"
    assert config.sw_model == "archie"
    assert config.lithology == "carbonate"
    assert config.skip_steps == ["bvw", "net_pay"]


def test_rock_names_with_archie_defaults_are_accepted_as_lithology():
    assert DynamicWorkflowConfig(lithology="shaly sand").lithology == "shaly sand"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rw": 0.0}, "rw must be greater than 0"),
        ({"a": -1.0}, "a must be greater than 0"),
        ({"vcl_cutoff": 40}, "vcl_cutoff must be a fraction between 0 and 1"),
        ({"bht_f": float("nan")}, "bht_f must be finite"),
        ({"rw": "0.05"}, "rw must be a number"),
        ({"apply_hc_correction": "yes"}, "apply_hc_correction must be True or False"),
        ({"gr_min": 120.0, "gr_max": 20.0}, "gr_min .* must be less than gr_max"),
        ({"lithology_csv": "a.csv", "lithology_intervals": "b.csv"}, "not both"),
    ],
)
def test_invalid_values_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DynamicWorkflowConfig(**kwargs)


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as excinfo:
        DynamicWorkflowConfig(hc_type="gass", rw=-1.0, skip_steps=["bvv"])
    message = str(excinfo.value)
    assert "hc_type" in message and "rw must" in message and "skip_steps" in message


def test_replace_revalidates():
    with pytest.raises(ValueError, match="hc_type"):
        dataclasses.replace(DynamicWorkflowConfig(), hc_type="gass")


def test_lithology_domain_error_names_the_setting_and_suggests_a_domain():
    with pytest.raises(ValueError) as info:
        DynamicWorkflowConfig(lithology_domain="carbonates")
    message = str(info.value)
    assert "lithology_domain='carbonates'" in message
    assert "auto, clastic, carbonate, evaporite" in message
    assert "did you mean 'carbonate'" in message
