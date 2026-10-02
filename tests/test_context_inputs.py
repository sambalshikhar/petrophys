"""LAS metadata inputs for the unified workflow."""

import numpy as np
import pytest

import petrophys as pp
from petrophys.workflow.zoned_batch import lithology_domain_from_metadata


def _well():
    well = pp.Well("HW-31")
    well.depth = np.array([1000.0, 1000.5, 1001.0, 1001.5])
    well.depth_unit = "m"
    return well


def test_lithology_domain_reads_round_trip_las_metadata():
    well = _well()
    assert lithology_domain_from_metadata(well) is None
    well.set_parameter("PP_DOMAIN", "Carbonate")
    assert lithology_domain_from_metadata(well) == "carbonate"
    well.set_parameter("PP_DOMAIN", "unknown")
    with pytest.raises(ValueError, match="PP_DOMAIN"):
        lithology_domain_from_metadata(well)


def test_interpret_uses_las_workflow_parameters_and_explicit_options_win():
    well = _well()
    well.set_parameter("A", 0.62)
    well.set_parameter("M", 2.15)
    well.set_parameter("N", 2.05)
    well.set_parameter("PP_DOMAIN", "carbonate")

    result = pp.interpret(well)
    assert result.config.a == pytest.approx(0.62)
    assert result.config.m == pytest.approx(2.15)
    assert result.config.n == pytest.approx(2.05)
    assert result.config.lithology_domain == "carbonate"

    overridden = pp.interpret(well, a=1.0, lithology_domain="clastic")
    assert overridden.config.a == pytest.approx(1.0)
    assert overridden.config.lithology_domain == "clastic"


def test_interpret_preserves_native_las_rw_provenance():
    well = _well()
    well.add_curve("GR", np.array([20.0, 30.0, 40.0, 50.0]), "API")
    well.set_parameter("RW", 0.07)
    well.set_parameter("RWT", 230.0)
    well.set_parameter("RWT__UNIT", "degF")

    result = pp.interpret(well)
    rw_step = next(step for step in result.steps_executed if step.step == "rw")

    assert result.config.rw is None
    assert rw_step.parameters["source_priority"] == "las.RW"
    assert rw_step.parameters["rw_at_measured_temp"] == pytest.approx(0.07)
    assert rw_step.parameters["measured_temp_f"] == pytest.approx(230.0)
