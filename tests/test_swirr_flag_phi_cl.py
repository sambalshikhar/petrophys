"""``DynamicWorkflowConfig.swirr_flag_phi_cl``: the PHIE behind the SWIRR_APP sample selection."""

import numpy as np
import pytest

import petrophys as pp
from petrophys.io import read_las
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.schema import describe_setting


def _permeability_step(result):
    return next(step for step in result.steps_executed if step.step == "permeability")


@pytest.fixture(scope="module")
def runs():
    """The GeoLink sample well 30/6-5 interpreted with each setting."""
    out = {}
    for choice in ("workflow", "all_shale_phit"):
        well = read_las(pp.sample("30_6-5"))
        out[choice] = DynamicWorkflow(DynamicWorkflowConfig(swirr_flag_phi_cl=choice)).run(well)
    return out


def test_default_is_the_reported_phie():
    assert DynamicWorkflowConfig().swirr_flag_phi_cl == "workflow"
    assert describe_setting("swirr_flag_phi_cl")["choices"] == ["workflow", "all_shale_phit"]


def test_choices_are_validated_and_normalized():
    assert DynamicWorkflowConfig(swirr_flag_phi_cl="ALL_SHALE_PHIT").swirr_flag_phi_cl == "all_shale_phit"
    with pytest.raises(ValueError, match=r"swirr_flag_phi_cl='all_shale'.*did you mean 'all_shale_phit'"):
        DynamicWorkflowConfig(swirr_flag_phi_cl="all_shale")


def test_workflow_choice_flags_reservoir_with_the_reported_phie(runs):
    flag = _permeability_step(runs["workflow"]).parameters["pre_permeability_resflag"]
    assert flag["phie_curve"] == "PHIE"
    assert "swirr_flag_phi_cl" not in flag


def test_all_shale_phit_changes_only_the_swirr_sample_selection(runs):
    base, legacy = runs["workflow"], runs["all_shale_phit"]
    flag = _permeability_step(legacy).parameters["pre_permeability_resflag"]
    phi_cl = flag["swirr_flag_phi_cl_value"]

    # phi_cl is the mean PHIT of the shale-lithology control samples, as reported
    assert 0.0 < phi_cl <= 0.6 and flag["swirr_flag_phi_cl_samples"] > 0
    assert f"phi_cl={phi_cl:.3f}" in flag["phie_curve"]
    assert any("recomputed with phi_cl=" in d for d in _permeability_step(legacy).decisions)

    # a different flag, so a different SWIRR_APP sample set
    base_flag = _permeability_step(base).parameters["pre_permeability_resflag"]
    assert flag["reservoir_samples"] != base_flag["reservoir_samples"]

    # the reported curves are untouched
    for curve in ("PHIT", "PHIE", "SW", "VCL"):
        np.testing.assert_array_equal(
            np.asarray(base.well.get(curve).data, dtype=float),
            np.asarray(legacy.well.get(curve).data, dtype=float),
        )
