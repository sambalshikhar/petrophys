"""The COSTA interpretation's step record is complete: the steps COSTA runs outside the workflow join the result under
the workflow's own step names, and the context trace keeps the package's names."""
import json
from pathlib import Path

import pytest

from petrophys.reporting._costa_context import (
    interpret_costa_context_las,
    interpret_costa_las,
    precomputed_step_results,
)

COSTA = Path(__file__).resolve().parents[1] / "src" / "petrophys" / "datasets" / "costa"
WELL = "HW-6"


@pytest.fixture(scope="module")
def context_items(tmp_path_factory):
    return interpret_costa_context_las(COSTA, tmp_path_factory.mktemp("costa"), well_names=[WELL])


def _steps(item):
    return {s.step: s for s in item.result.base_result.steps_executed}


def test_the_record_names_the_steps_that_wrote_vcl_the_minerals_and_perm(context_items):
    (item,) = context_items
    assert item.success
    steps = _steps(item)
    assert "VCL" in steps["vclay"].output_curves and "GR" in steps["vclay"].input_curves
    assert {"VCALC", "VDOLO"} <= set(steps["lithology"].output_curves)
    assert steps["permeability"].output_curves == ["PERM"]
    assert not [s for s in steps if s.startswith("costa_")]


def test_curves_the_input_carries_are_not_given_a_producing_step(context_items):
    (item,) = context_items
    produced = {c for s in item.result.base_result.steps_executed for c in s.output_curves}
    assert "core_correction" not in _steps(item)                      # PHIE_CC and PERM_CC come with the LAS
    assert not produced & {"PHIE_CC", "PERM_CC"} & set(item.input_curves)


def test_each_curve_has_at_most_one_producing_step_from_outside_the_workflow(context_items):
    (item,) = context_items
    outside = [s for s in item.result.base_result.steps_executed if s.step in ("vclay", "lithology", "permeability", "crossover_flag")]
    workflow = {c for s in item.result.base_result.steps_executed if s not in outside for c in s.output_curves}
    for s in outside:
        assert not set(s.output_curves) & workflow, s.step
        assert s.step not in item.result.base_result.steps_skipped


def test_the_context_trace_keeps_the_package_names(tmp_path):
    (item,) = interpret_costa_las(COSTA, tmp_path, well_names=[WELL])
    payload = json.loads(Path(item.json_log_file).read_text())
    assert [s["step"] for s in payload["costa_precomputed_steps"]][:2] == ["costa_core_curve_aliases", "costa_vcl"]
    assert "vclay" in _steps(item)                                    # and the returned result has the record


def test_precomputed_step_results_skip_what_the_input_or_the_workflow_wrote(context_items):
    from petrophys.reporting._costa_context import load_costa_parameters
    (item,) = context_items
    well, params = item.result.well, load_costa_parameters()[WELL]
    every = {c for c in well.curves}
    assert precomputed_step_results(well, params, every, []) == []        # everything an input: nothing to record
    names = {r.step for r in precomputed_step_results(well, params, item.input_curves, ["VCL"])}
    assert "vclay" not in names and "lithology" in names                    # VCL already written by a workflow step
