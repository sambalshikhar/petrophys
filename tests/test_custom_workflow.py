"""Custom workflows: building, validation, running, results, sharing and batches."""

import copy
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys.workflow import (
    Step,
    Workflow,
    WorkflowBatchResult,
    WorkflowRunResult,
    WorkflowStepError,
    WorkflowValidationError,
    steps,
    workflow_step,
)
from tests import custom_workflow_steps as helpers

BRENT = (2841.0, 2958.0)

# 30/6-5 stores core saturation (SWC, SOC) in percent although it is labelled a
# fraction; every standardization pass corrects it and says so. Not under test.
pytestmark = pytest.mark.filterwarnings("ignore:.*inferred percent core saturation:UserWarning")


def _read(name, **options):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*inferred percent core saturation")
        return pp.read(pp.sample(name), **options)


@pytest.fixture(scope="module")
def well():
    return _read("30_6-5", formations=pp.sample("formations_30_6-5"))


@pytest.fixture(scope="module")
def reference(well):
    """pp.interpret on 30/6-5 with the documented Rw."""
    return pp.interpret(well, rw=0.07, rw_meas_temp_f=230)


def quick_look():
    return Workflow("Brent quick look", [
        steps.clay_volume(method="gr", gr_clean=20, gr_shale=110),
        steps.porosity(method="density", matrix_density=2.65),
        steps.formation_water(rw=0.07, rw_temp_f=230),
        steps.saturation(model="archie", a=1, m=2, n=2),
        steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.10, sw_cutoff=0.60),
    ])


def same(a, b):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


# -- building -------------------------------------------------------------------


def test_top_level_exports():
    assert pp.Workflow is Workflow
    assert pp.workflow_step is workflow_step
    assert "Workflow" in pp.__all__ and "workflow_step" in pp.__all__
    assert set(steps.__all__) >= {
        "standardize", "temperature", "clay_volume", "porosity", "formation_water",
        "saturation", "permeability", "rock_quality", "bvw", "mhi", "net_pay",
        "spectral_gamma", "interpret", "formation_summary",
    }


def test_add_chains_and_keeps_order():
    wf = Workflow("demo")
    assert wf.add(steps.clay_volume()) is wf
    wf.add(steps.porosity()).add(steps.formation_water(rw=0.07, rw_temp_f=230))
    assert [step.name for step in wf] == ["clay_volume", "porosity", "formation_water"]
    assert len(wf) == 3 and "porosity" in wf and "saturation" not in wf
    assert wf[1] is wf["porosity"]
    assert isinstance(wf.steps, tuple)


def test_steps_record_explicit_params_as_typed():
    step = steps.clay_volume(method="GR", gr_clean=20, gr_shale=110.5)
    assert step.params == {"method": "gr", "gr_clean": 20, "gr_shale": 110.5}
    assert isinstance(step.params["gr_clean"], int)
    assert repr(step) == "steps.clay_volume(method='gr', gr_clean=20, gr_shale=110.5)"
    assert step.name == step.kind == "clay_volume"
    named = steps.clay_volume(method="sp", name="vcl_sp")
    assert named.name == "vcl_sp" and repr(named).endswith("name='vcl_sp')")
    assert steps.porosity().params == {}


def test_step_contract_attributes():
    step = steps.saturation(model="indonesia", rcl=4.0)
    assert step.requires == ("PHIE", "RDEP or RT", "RW", "VCL")
    assert step.produces == {"SW": "v/v", "SXO": "v/v"}
    assert step.description
    assert not step.is_custom
    assert step.to_dict() == {"name": "saturation", "kind": "saturation",
                              "params": {"model": "indonesia", "rcl": 4.0}}


def test_with_params_revalidates_and_keeps_name():
    step = steps.saturation(model="archie", n=2.0, name="sw")
    changed = step.with_params(n=2.2)
    assert changed.params["n"] == 2.2 and changed.name == "sw"
    assert step.params["n"] == 2.0
    with pytest.raises(ValueError, match="n must be greater than 0"):
        step.with_params(n=-1)


def test_add_rejects_uncalled_factories_and_plain_functions():
    with pytest.raises(TypeError, match=r"steps.porosity is a step factory; call it"):
        Workflow("x").add(steps.porosity)
    with pytest.raises(TypeError, match="linear_vsh is a step factory"):
        Workflow("x").add(helpers.linear_vsh)
    with pytest.raises(TypeError, match="decorate it with @petrophys.workflow_step"):
        Workflow("x").add(helpers.not_a_step)


def test_duplicate_names_are_rejected_with_a_fix():
    wf = Workflow("x", [steps.porosity()])
    with pytest.raises(WorkflowValidationError, match="used twice.*name='porosity_2'"):
        wf.add(steps.porosity(method="density"))
    wf.add(steps.porosity(method="density", name="porosity_density"))
    assert len(wf) == 2


def test_forward_dependency_is_caught_when_adding():
    wf = Workflow("x", [steps.saturation()])
    with pytest.raises(WorkflowValidationError,
                       match="'saturation' needs RW, which only the later step 'formation_water'"):
        wf.add(steps.formation_water(rw=0.07, rw_temp_f=230))
    assert [step.name for step in wf] == ["saturation"]
    wf.insert(steps.formation_water(rw=0.07, rw_temp_f=230), before="saturation")
    assert [step.name for step in wf] == ["formation_water", "saturation"]


def test_insert_replace_remove_copy():
    wf = Workflow("x", [steps.clay_volume(), steps.porosity()])
    wf.insert(steps.temperature(), before="clay_volume")
    wf.insert(steps.formation_water(rw=0.07, rw_temp_f=230), after="porosity")
    assert [s.name for s in wf] == ["temperature", "clay_volume", "porosity", "formation_water"]
    wf.replace("clay_volume", steps.clay_volume(method="minimum"))
    assert wf["clay_volume"].params == {"method": "minimum"}
    clone = wf.copy("clone")
    clone.remove("temperature")
    assert len(wf) == 4 and len(clone) == 3 and clone.name == "clone"
    with pytest.raises(KeyError, match="did you mean 'porosity'"):
        wf.remove("porosty")
    with pytest.raises(TypeError, match="exactly one of before"):
        wf.insert(steps.bvw())
    with pytest.raises(WorkflowValidationError):
        wf.replace("porosity", steps.clay_volume(method="sp"))  # duplicate name


def test_describe_table():
    table = quick_look().describe()
    assert list(table.columns) == ["Step", "Kind", "Requires", "Produces", "Parameters"]
    assert table.index.tolist() == [1, 2, 3, 4, 5]
    row = table.loc[2]
    assert row["Requires"] == "RHOB, VCL"
    assert row["Produces"] == "PHID, PHIT, PHIE"
    assert row["Parameters"] == "method='density', matrix_density=2.65"
    assert table.loc[3, "Requires"] == "-"
    assert Workflow("empty").describe().empty


def test_repr_and_html():
    wf = quick_look()
    text = repr(wf)
    assert text.splitlines()[0] == "Workflow('Brent quick look', 5 steps)"
    assert "1. clay_volume" in text and "steps.net_pay(vcl_cutoff=0.35" in text
    html = wf._repr_html_()
    assert "<table" in html and "Brent quick look" in html
    assert "0 steps" in Workflow("empty")._repr_html_()


# -- parameter validation -------------------------------------------------------


def test_invalid_choice_suggests_the_closest_option():
    with pytest.raises(ValueError) as info:
        steps.porosity(method="densty")
    message = str(info.value)
    assert message.startswith("Invalid steps.porosity() parameters:")
    assert "use one of: density, neutron_density, neutron, sonic (did you mean 'density'?)" in message


@pytest.mark.parametrize("call, suggestion", [
    (lambda: steps.net_pay(vcl=0.35), "vcl_cutoff"),
    (lambda: steps.net_pay(vcl="VSH"), "vcl_curve"),
    (lambda: steps.saturation(nn=2), "n"),
    (lambda: steps.formation_water(rw=0.07, rw_temp=230), "rw_temp_f"),
])
def test_unknown_parameter_names_suggest_a_valid_one(call, suggestion):
    with pytest.raises(TypeError, match=f"did you mean '{suggestion}'"):
        call()


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as info:
        steps.clay_volume(method="gr", gr_clean=120, gr_shale=20, gr_transform="larionov")
    message = str(info.value)
    assert "gr_clean (120) must be less than gr_shale (20)" in message
    assert "gr_transform='larionov' is not valid" in message
    assert len(message.splitlines()) == 3


@pytest.mark.parametrize("call, message", [
    (lambda: steps.saturation(model="archie", rcl=4), "rcl is not used by model='archie'"),
    (lambda: steps.clay_volume(method="sp", gr_clean=20), "gr_clean is not used by method='sp'"),
    (lambda: steps.porosity(method="sonic", matrix_density=2.7),
     "matrix_density is not used by method='sonic'"),
    (lambda: steps.saturation(model="waxman_smits"), "needs cec="),
    (lambda: steps.formation_water(rw=0.07), "rw_temp_f is required with rw"),
    (lambda: steps.formation_water(), "give rw="),
    (lambda: steps.formation_water(rw=0.07, rw_temp_f=230, salinity_ppm=3e4), "not both"),
    (lambda: steps.formation_water(rw="headers"), "'header'"),
    (lambda: steps.porosity(clay_porosity=1.5), "clay_porosity must be a fraction"),
    (lambda: steps.porosity(matrix="sandston"), "did you mean 'sandstone'"),
    (lambda: steps.temperature(bht_f=200, gradient_f_per_100ft=1.5), "not both"),
    (lambda: steps.permeability(method="timur", swirr=0), "between 0 and 1"),
    (lambda: steps.permeability(method="lith_vcl", swirr=0.2), "swirr is not used"),
    (lambda: steps.net_pay(vcl_curve=None, vcl_cutoff=0.3), "vcl_cutoff needs a clay volume curve"),
    (lambda: steps.spectral_gamma(api_coefficients={"POTASIUM": 16}), "did you mean 'POTASSIUM'"),
    (lambda: steps.formation_summary(formations=pd.DataFrame()), "in-memory tables"),
])
def test_invalid_values_are_explained(call, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        call()


def test_interpret_step_validates_settings():
    with pytest.raises(TypeError, match="did you mean 'sw_model'"):
        steps.interpret(sw_modle="archie")
    with pytest.raises(ValueError, match=r"Invalid steps.interpret\(\) settings"):
        steps.interpret(sw_model="archi")
    with pytest.raises(TypeError, match="cannot be saved with the workflow"):
        steps.interpret(peer_wells=[pp.Well("offset")])


def test_custom_step_parameters_are_checked():
    with pytest.raises(TypeError, match="did you mean 'gr_clean'"):
        helpers.linear_vsh(gr_clen=25)
    with pytest.raises(TypeError, match="cannot be saved with the workflow"):
        helpers.linear_vsh(gr_clean=np.array([20.0]))
    step = helpers.linear_vsh(25)
    assert step.params == {"gr_clean": 25}


# -- custom steps -----------------------------------------------------------------


def test_workflow_step_builds_configured_steps():
    step = helpers.linear_vsh(gr_clean=25)
    assert isinstance(step, Step) and step.is_custom
    assert step.kind == "tests.custom_workflow_steps:linear_vsh"
    assert step.name == "linear_vsh"
    assert step.requires == ("GR",) and step.produces == {"VSH_LIN": "v/v"}
    assert step.description == "Linear shale volume from gamma ray."
    assert repr(step) == "linear_vsh(gr_clean=25)"
    assert "requires GR" in repr(helpers.linear_vsh)


def test_workflow_step_signature_rules():
    with pytest.raises(TypeError, match="must take the well as its first argument"):
        workflow_step(lambda: None)
    with pytest.raises(TypeError, match="reserved for the step name"):
        @workflow_step
        def named(well, name="x"):
            return None
    with pytest.raises(TypeError, match=r"cannot take \*args"):
        @workflow_step
        def variadic(well, *args):
            return None
    with pytest.raises(TypeError, match="unit string"):
        workflow_step(produces={"X": 1})

    @workflow_step
    def bare(well):
        """Does nothing."""

    assert bare().requires == () and bare().produces == {}


def test_custom_step_adds_curves_with_units(well):
    run = Workflow("x", [helpers.linear_vsh(gr_clean=25, gr_shale=125)]).run(well)
    curve = run.well.get("VSH_LIN")
    assert curve.unit == "v/v"
    gr = well.get("GR").data
    assert same(curve.data, np.clip((gr - 25) / 100, 0, 1))
    trace = run.steps[0]
    assert trace.method == "tests.custom_workflow_steps.linear_vsh"
    assert trace.input_curves == ["GR"] and trace.output_curves == ["VSH_LIN"]
    assert trace.parameters == {"gr_clean": 25, "gr_shale": 125}


@pytest.mark.parametrize("returned, message", [
    (lambda well: {"X": np.zeros(5)}, "returned X with 5 values; the well has"),
    (lambda well: {"X": np.zeros(well.n_samples), "Y": np.zeros(well.n_samples)},
     "returned Y, which is not declared"),
    (lambda well: {}, "did not return X"),
    (lambda well: None, "returned None, but declares"),
    (lambda well: 3.0, "must return a dict of curves"),
])
def test_custom_step_output_contract(well, returned, message):
    def compute(well):
        return returned(well)

    compute.__qualname__ = "compute"
    factory = workflow_step(compute, produces={"X": "v/v"})
    with pytest.raises(WorkflowStepError, match=message) as info:
        Workflow("x", [factory()]).run(well)
    assert info.value.step == "compute"


def test_custom_step_table_becomes_an_artifact(well):
    run = Workflow("x", [helpers.gr_histogram(bins=3)]).run(well)
    table = run.artifacts["gr_histogram"]
    assert list(table.columns) == ["low", "high", "count"] and len(table) == 3
    assert run.steps[0].output_curves == []


def test_replacing_an_input_curve_is_recorded(well):
    @workflow_step(requires=["GR"], produces={"GR": "gAPI"})
    def smooth_gr(well, window=5):
        values = pd.Series(well.get("GR").data).rolling(window, center=True, min_periods=1)
        return {"GR": values.median().to_numpy()}

    run = Workflow("x", [smooth_gr()]).run(well)
    assert "Replaced input curve GR" in run.steps[0].decisions
    assert not same(run.well.get("GR").data, well.get("GR").data)


# -- check and strict/non-strict runs ------------------------------------------------


def test_check_counts_earlier_outputs(well):
    table = quick_look().check(well)
    assert list(table.columns) == ["Step", "Requires", "Met", "Source", "Note"]
    assert table["Met"].all()
    sources = dict(zip(table["Step"] + ":" + table["Requires"], table["Source"]))
    assert sources["porosity:VCL"] == "step clay_volume"
    assert sources["saturation:RDEP or RT"] == "well (RDEP)"
    assert sources["net_pay:SW"] == "step saturation"
    assert (table.loc[table["Step"] == "formation_water", "Requires"] == "(nothing)").all()


def test_check_explains_missing_inputs_and_their_consequences(well):
    wf = Workflow("gaps", [steps.porosity(method="density"), steps.saturation(), steps.bvw()])
    table = wf.check(well)
    missing = table[~table["Met"]]
    notes = dict(zip(missing["Step"] + ":" + missing["Requires"], missing["Note"]))
    assert notes["porosity:VCL"] == ("needs VCL (clay volume), which is not on this well; "
                                     "add steps.clay_volume() before 'porosity'")
    assert "step 'porosity' would produce but cannot run" in notes["saturation:PHIE"]
    assert "steps.formation_water(rw=..., rw_temp_f=...)" in notes["saturation:RW"]
    assert "step 'saturation' would produce but cannot run" in notes["bvw:SW"]


def test_check_distinguishes_empty_curves_from_missing_ones():
    costa = _read("HW-31")  # SW_INPUT exists but holds no valid values
    table = Workflow("x", [steps.bvw(phie_curve="PHIE_CC", sw_curve="SW_INPUT")]).check(costa)
    assert table.loc[~table["Met"], "Note"].tolist() == [
        "needs SW_INPUT (water saturation), which has no valid samples on this well"]


def test_run_accepts_a_file_path():
    run = Workflow("x", [steps.clay_volume()]).run(pp.sample("HW-3"))
    assert run.well.name == "HW-3" and run.status == "complete"


def test_strict_run_raises_before_any_calculation(well):
    wf = Workflow("gaps", [steps.clay_volume(), steps.saturation(), steps.net_pay()])
    target = copy.deepcopy(well)
    curves = list(target.curves)
    with pytest.raises(WorkflowValidationError) as info:
        wf.run(target, inplace=True)
    message = str(info.value)
    assert message.startswith("Workflow 'gaps' cannot run on well '30/6-5':")
    assert "  - saturation: needs PHIE" in message and "  - saturation: needs RW" in message
    assert "  - net_pay: needs PHIE" in message and "  - net_pay: needs SW" in message
    assert "pass strict=False" in message
    assert {problem["step"] for problem in info.value.problems} == {"saturation", "net_pay"}
    assert list(target.curves) == curves  # nothing ran, even in place


def test_non_strict_run_skips_steps_with_reasons(well):
    wf = Workflow("gaps", [steps.clay_volume(), steps.saturation(), steps.net_pay(),
                           helpers.linear_vsh()])
    run = wf.run(well, strict=False)
    assert run.status == "partial" and run.success
    assert [trace.step for trace in run.steps] == ["clay_volume", "linear_vsh"]
    assert set(run.steps_skipped) == {"saturation", "net_pay"}
    assert run.steps_skipped["saturation"].startswith("Needs PHIE (effective porosity)")
    assert "step 'saturation' would produce but cannot run" in run.steps_skipped["net_pay"]
    table = run.to_dataframe()
    assert table["Status"].tolist() == ["OK", "SKIPPED", "SKIPPED", "OK"]


def test_failed_step_strict_and_non_strict(well):
    wf = Workflow("x", [helpers.failing_step(), helpers.linear_vsh()])
    with pytest.raises(WorkflowStepError, match="Step 'failing_step' .* deliberate failure") as info:
        wf.run(well)
    assert isinstance(info.value.__cause__, RuntimeError)
    run = wf.run(well, strict=False)
    assert run.status == "failed" and not run.success
    assert run.errors == ["failing_step: deliberate failure"]
    assert not run.steps[0].success and "deliberate failure" in run.steps[0].message
    assert run.steps[1].success
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run.raise_for_errors()


def test_step_after_a_failed_producer_is_skipped(well):
    @workflow_step(requires=["GR"], produces={"PHIE": "v/v", "SW": "v/v"})
    def broken_petrophysics(well):
        raise ValueError("no luck")

    wf = Workflow("x", [broken_petrophysics(), helpers.hydrocarbon_fraction()])
    run = wf.run(well, strict=False)
    assert run.steps_skipped["hydrocarbon_fraction"] == (
        "Needs PHIE, which step 'broken_petrophysics' did not produce; "
        "needs SW, which step 'broken_petrophysics' did not produce"
    )


def test_run_leaves_the_input_well_unchanged(well):
    curves = list(well.curves)
    run = quick_look().run(well)
    assert list(well.curves) == curves
    assert run.well is not well
    assert {"VCL", "PHIE", "RW", "SW", "NET_PAY"} <= set(run.well.curves)


def test_one_off_parameter_overrides(well):
    wf = quick_look()
    run = wf.run(well, params={"saturation": {"n": 2.5}, "net_pay": {"sw_cutoff": 0.5}})
    assert run.workflow["saturation"].params["n"] == 2.5
    assert wf["saturation"].params["n"] == 2
    base = wf.run(well)
    assert not same(run.well.get("SW").data, base.well.get("SW").data)
    with pytest.raises(KeyError, match="did you mean 'saturation'"):
        wf.run(well, params={"saturaton": {"n": 2.2}})
    with pytest.raises(TypeError, match="did you mean 'n'"):
        wf.run(well, params={"saturation": {"nn": 2.2}})
    with pytest.raises(TypeError, match="must be a dict of parameter values"):
        wf.run(well, params={"saturation": 2.2})


# -- results ---------------------------------------------------------------------------


def test_run_result_trace(well):
    run = quick_look().run(well)
    assert isinstance(run, WorkflowRunResult)
    assert run.status == "complete" and run.steps_skipped == {} and run.errors == []
    clay, porosity, water, saturation, pay = run.steps
    assert clay.method == "gr_linear" and clay.output_curves == ["VCL"]
    assert clay.parameters["gr_clean_used"] == 20.0
    assert porosity.input_curves == ["RHOB", "VCL"]
    assert porosity.output_curves == ["PHID", "PHIT", "PHIE"]
    assert "PHID = (2.65 - RHOB) / (2.65 - 1)" in porosity.decisions
    assert water.method == "given_rw_arps" and water.parameters["rw_used"] == 0.07
    assert saturation.method == "archie" and saturation.input_curves == ["PHIE", "RW", "RDEP"]
    assert pay.parameters["sw_cutoff_used"] == 0.6
    assert run.output_curves[:2] == ["VCL", "PHID"]
    assert run.well.get("SW").metadata.custom["workflow_step"] == "saturation"
    assert repr(run) == "<WorkflowRunResult 'Brent quick look' on '30/6-5': complete, 5 run, 0 skipped>"
    assert "<table" in run._repr_html_()


def test_summary_over_a_window(well):
    run = quick_look().run(well)
    summary = run.summary(top=BRENT[0], bottom=BRENT[1])
    assert summary["status"] == "complete" and summary["workflow"] == "Brent quick look"
    assert list(summary["key_results"]) == ["VCL", "PHID", "PHIT", "PHIE", "RW", "SW", "SXO"]
    depth = well.depth
    inside = (depth >= BRENT[0]) & (depth <= BRENT[1])
    assert summary["key_results"]["SW"]["mean"] == pytest.approx(
        np.nanmean(run.well.get("SW").data[inside]))
    pay = np.count_nonzero(run.well.get("NET_PAY").data[inside] == 1) * well.step
    assert summary["net_thickness"]["net_pay"] == round(pay, 2)
    assert summary["executed_steps"][0] == {"name": "clay_volume", "kind": "clay_volume",
                                            "method": "gr_linear", "success": True}
    json.dumps(summary)


def test_formation_summary_matches_the_prebuilt_result_table(well, reference):
    run = Workflow("x", [steps.interpret(rw=0.07, rw_meas_temp_f=230)]).run(well)
    table = run.formation_summary(rank="formation")
    expected = reference.formation_summary(rank="formation")
    pd.testing.assert_frame_equal(table, expected)
    with pytest.raises(ValueError, match="No formations are attached"):
        Workflow("x", [steps.clay_volume()]).run(_read("HW-3")).formation_summary()


def test_formation_summary_step_stores_a_table(well):
    wf = quick_look().add(steps.formation_summary(rank="formation"))
    run = wf.run(well)
    table = run.artifacts["formation_summary"]
    assert "Etive Fm." in table["Formation"].tolist()
    pd.testing.assert_frame_equal(table, run.formation_summary(rank="formation"))
    check = wf.check(_read("30_6-5"))
    row = check[check["Step"] == "formation_summary"].iloc[0]
    assert not row["Met"] and "formations=" in row["Note"]


def test_print_trace(well, capsys):
    run = Workflow("gaps", [steps.clay_volume(), steps.saturation()]).run(well, strict=False)
    run.print_trace()
    text = capsys.readouterr().out
    assert "WORKFLOW TRACE: gaps on 30/6-5" in text
    assert "[OK  ] clay_volume  (gr_linear)" in text
    assert "[SKIP] saturation: Needs PHIE" in text


def test_json_export(well, tmp_path):
    run = quick_look().add(helpers.gr_histogram()).run(well)
    text = run.to_json(include_curves=["SW"])
    data = json.loads(text)
    assert data["schema"] == "custom_workflow_trace"
    assert data["workflow"]["steps"][0] == {"name": "clay_volume", "kind": "clay_volume",
                                           "params": {"method": "gr", "gr_clean": 20,
                                                      "gr_shale": 110}}
    assert data["steps"][3]["step"] == "saturation" and data["steps"][3]["kind"] == "saturation"
    assert data["artifacts"]["gr_histogram"]["type"] == "table"
    assert len(data["artifacts"]["gr_histogram"]["records"]) == 4
    assert data["curve_summaries"]["SW"]["unit"] == "v/v"
    path = run.to_json(tmp_path / "trace.json")
    assert json.loads(path.read_text())["status"] == "complete"


def test_las_export_carries_the_workflow(well, tmp_path):
    lasio = pytest.importorskip("lasio")
    run = quick_look().run(well)
    path = run.to_las(tmp_path / "out.las")
    las = lasio.read(path)
    params = {item.mnemonic: item for item in las.params}
    assert params["PP_WORKFLOW"].value == "Brent quick look"
    assert params["PP_STATUS"].value == "complete"
    assert params["PP_STEP01"].value == "clay_volume"
    assert params["PP_STEP01"].descr == "steps.clay_volume(method='gr', gr_clean=20, gr_shale=110) [run]"
    definition = las.other.split("Workflow definition (JSON):")[1].strip()
    assert Workflow.from_dict(json.loads(definition)) == quick_look()
    assert "SW" in las.curves.keys()


def test_plotting(well):
    run = quick_look().run(well)
    viewer = pp.plot(run.well, depth_range=BRENT)
    viewer.close()
    viewer = run.plot(depth_range=BRENT, template="auto")
    assert viewer.fig is not None
    viewer.close()


# -- saving and loading ---------------------------------------------------------------


def test_save_and_load_round_trip_with_a_custom_step(tmp_path, well):
    wf = quick_look().add(helpers.linear_vsh(gr_clean=25))
    wf.description = "Quick look plus linear Vsh"
    path = wf.save(tmp_path / "nested" / "brent.json")
    data = json.loads(path.read_text())
    assert data["schema"] == "petrophys.workflow" and data["schema_version"] == 1
    assert data["petrophys_version"] == pp.__version__
    assert data["steps"][-1] == {"name": "linear_vsh",
                                 "kind": "tests.custom_workflow_steps:linear_vsh",
                                 "params": {"gr_clean": 25}}
    loaded = Workflow.load(path)
    assert loaded == wf and loaded.description == "Quick look plus linear Vsh"
    assert Workflow.from_dict(wf.to_dict()) == wf
    first, second = wf.run(well), loaded.run(well)
    for curve in ("VCL", "PHIE", "SW", "NET_PAY", "VSH_LIN"):
        assert same(first.well.get(curve).data, second.well.get(curve).data)


def test_loading_reports_unusable_steps(tmp_path):
    data = quick_look().to_dict()
    bad_module = copy.deepcopy(data)
    bad_module["steps"][0] = {"name": "vsh", "kind": "missing_module:vsh", "params": {}}
    with pytest.raises(ValueError, match="Step 1 \\('vsh'\\): cannot import custom step "
                                         "'missing_module:vsh'.*make the module importable"):
        Workflow.from_dict(bad_module)
    plain = copy.deepcopy(data)
    plain["steps"][0]["kind"] = "tests.custom_workflow_steps:not_a_step"
    with pytest.raises(ValueError, match="is not a workflow step"):
        Workflow.from_dict(plain)
    unknown = copy.deepcopy(data)
    unknown["steps"][0]["kind"] = "clay_volumes"
    with pytest.raises(ValueError, match="unknown built-in step 'clay_volumes' \\(did you mean 'clay_volume'\\?\\)"):
        Workflow.from_dict(unknown)
    invalid = copy.deepcopy(data)
    invalid["steps"][1]["params"]["method"] = "densty"
    with pytest.raises(ValueError, match="Step 2 \\('porosity'\\) could not be rebuilt"):
        Workflow.from_dict(invalid)
    newer = dict(data, schema_version=99)
    with pytest.raises(ValueError, match="Upgrade petrophys"):
        Workflow.from_dict(newer)
    with pytest.raises(ValueError, match="Not a petrophys workflow"):
        Workflow.from_dict({"schema": "other"})
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="broken.json is not valid JSON"):
        Workflow.load(broken)
    with pytest.raises(FileNotFoundError):
        Workflow.load(tmp_path / "missing.json")


def test_saving_steps_that_cannot_be_imported_again(tmp_path):
    def build():
        @workflow_step(requires=["GR"], produces={"X": ""})
        def local(well):
            return {"X": well.get("GR").data}
        return local

    with pytest.raises(ValueError, match="defined inside a function"):
        Workflow("x", [build()()]).save(tmp_path / "x.json")

    def main_step(well):
        return None

    main_step.__module__, main_step.__qualname__ = "__main__", "main_step"
    with pytest.warns(UserWarning, match="defined in __main__"):
        Workflow("x", [workflow_step(main_step)()]).save(tmp_path / "y.json")


# -- many wells ------------------------------------------------------------------------


def test_run_many_collects_failures_per_well(tmp_path):
    broken = tmp_path / "broken.las"
    broken.write_text("not a LAS file\n")
    wf = Workflow("carbonate", [
        steps.clay_volume(method="gr", gr_clean=10, gr_shale=80),
        steps.porosity(matrix="limestone"),
        steps.formation_water(rw="header"),
        steps.saturation(),
        steps.net_pay(lithology="carbonate"),
    ])
    batch = wf.run_many([pp.sample("HW-3"), broken, pp.sample("HW-4")],
                        formations=pp.sample("formations_costa"))
    assert isinstance(batch, WorkflowBatchResult)
    assert batch.wells == ["HW-3", "broken.las", "HW-4"]
    assert list(batch.results) == ["HW-3", "HW-4"] and list(batch.errors) == ["broken.las"]
    assert batch["HW-3"].status == "complete"
    with pytest.raises(KeyError, match="did not run"):
        batch["broken.las"]
    with pytest.raises(RuntimeError, match="1 of 3 wells failed"):
        batch.raise_for_errors()

    table = batch.summary_table()
    assert table["Well"].tolist() == ["HW-3", "broken.las", "HW-4"]
    assert table["Status"].tolist() == ["complete", "error", "complete"]
    assert table.loc[0, "Steps Run"] == 5 and table.loc[0, "Depth Unit"] == "ft"
    assert table.loc[1, "Error"].startswith(("ValueError", "OSError", "Exception", "LASHeaderError",
                                             "KeyError", "IndexError"))
    run = batch["HW-3"]
    pay = np.count_nonzero(run.well.get("NET_PAY").data == 1) * run.well.step
    assert table.loc[0, "Net Pay"] == pytest.approx(pay)

    by_formation = batch.summary_table(by_formation=True)
    assert by_formation.loc[0, "Formation"] == "KHARAIB-2"
    assert by_formation.loc[0, "Net Pay"] == pytest.approx(pay)
    expected = run.formation_summary().iloc[0]
    assert by_formation.loc[0, "Avg PHIE"] == pytest.approx(expected["Avg PHIE"], nan_ok=True)


def test_run_many_reads_a_folder_and_accepts_wells(tmp_path):
    for name in ("HW-3", "HW-4"):
        shutil.copy(pp.sample(name), tmp_path / f"{name}.las")
    (tmp_path / "notes.txt").write_text("ignored")
    wf = Workflow("vcl", [steps.clay_volume()])
    batch = wf.run_many(tmp_path)
    assert batch.wells == ["HW-3", "HW-4"] and not batch.errors
    well = _read("HW-5")
    curves = list(well.curves)
    again = wf.run_many([well, well])
    assert again.wells == ["HW-5", "HW-5 (2)"]
    assert list(well.curves) == curves
    with pytest.raises(FileNotFoundError, match="No files matching"):
        wf.run_many(tmp_path, pattern="*.dlis")
    with pytest.raises(ValueError, match="at least one well"):
        wf.run_many([])


def test_run_many_strict_and_non_strict():
    wf = Workflow("needs RW", [steps.clay_volume(), steps.porosity(), steps.saturation()])
    strict = wf.run_many([pp.sample("HW-3")])
    assert "WorkflowValidationError" in strict.errors["HW-3"]
    relaxed = wf.run_many([pp.sample("HW-3")], strict=False)
    assert relaxed["HW-3"].status == "partial"
    assert relaxed.summary_table().loc[0, "Skipped Steps"] == "saturation"


# -- equivalence with pp.interpret and the calculation modules on 30/6-5 --------------------


def test_interpret_step_matches_pp_interpret(well, reference):
    run = Workflow("prebuilt", [steps.interpret(rw=0.07, rw_meas_temp_f=230)]).run(well)
    for curve in ("VCL", "PHIT", "PHIE", "SW", "PERM", "NET_PAY", "RW", "TEMP"):
        assert same(run.well.get(curve).data, reference.well.get(curve).data), curve
    artifact = run.artifacts["interpret"]
    assert artifact.status == reference.status
    assert run.summary(*BRENT)["net_thickness"] == reference.summary(*BRENT)["net_thickness"]
    assert "vclay: linear" in run.steps[0].decisions


def test_interpret_step_extended_with_custom_steps(well, reference):
    wf = Workflow("extended", [
        helpers.linear_vsh(),
        steps.interpret(rw=0.07, rw_meas_temp_f=230),
        helpers.hydrocarbon_fraction(),
    ])
    run = wf.run(well)
    expected = reference.well.get("PHIE").data * (1 - reference.well.get("SW").data)
    assert same(run.well.get("HCPV_FRAC").data, expected)
    assert same(run.well.get("SW").data, reference.well.get("SW").data)
    assert wf.check(well)["Met"].all()


def test_clay_volume_matches_vcl_module(well):
    from petrophys.interpretation import VclModule

    run = Workflow("x", [steps.clay_volume(method="gr", gr_clean=15, gr_shale=120)]).run(well)
    expected = VclModule(gr_min=15, gr_max=120, use_sp=False, use_nd=False).calculate(well).vcl
    assert same(run.well.get("VCL").data, expected)


def test_clay_volume_minimum_writes_each_indicator(well):
    run = Workflow("x", [steps.clay_volume(method="minimum")]).run(well)
    data = run.well
    stacked = np.vstack([data.get(name).data for name in ("VCL_GR", "VCL_SP", "VCL_ND")])
    assert same(data.get("VCL").data, np.nanmin(stacked, axis=0))
    assert run.steps[0].method == "minimum_gr_sp_nd"


def test_temperature_and_formation_water_match_pp_interpret(well, reference):
    wf = Workflow("x", [steps.temperature(), steps.formation_water(rw=0.07, rw_temp_f=230)])
    run = wf.run(well)
    assert same(run.well.get("TEMP").data, reference.well.get("TEMP").data)
    assert same(run.well.get("RW").data, reference.well.get("RW").data)
    no_temp = Workflow("x", [steps.formation_water(rw=0.07, rw_temp_f=230)]).run(well)
    assert same(no_temp.well.get("RW").data, reference.well.get("RW").data)


def test_formation_water_from_the_las_header():
    from petrophys.interpretation.electrical.rw import correct_rw_temperature

    costa = _read("HW-3")
    run = Workflow("x", [steps.temperature(), steps.formation_water(rw="header")]).run(costa)
    expected = correct_rw_temperature(0.0682, 68.0, run.well.get("TEMP").data, "F")
    assert same(run.well.get("RW").data, expected)
    well = _read("30_6-5")  # has RW in its header but no RWT
    table = Workflow("x", [steps.formation_water(rw="header")]).check(well)
    assert not table["Met"].iloc[0] and "RWT" in table["Note"].iloc[0]
    run = Workflow("x", [steps.formation_water(rw="header", rw_temp_f=75)]).run(well)
    assert run.steps[0].parameters["rw_used"] == 0.7


def test_porosity_matches_pp_interpret_with_a_fixed_matrix(well):
    reference = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, lithology_source="config",
                             matrix_type="sandstone", lithology_domain="clastic")
    wf = Workflow("x", [steps.standardize(), steps.clay_volume(),
                        steps.porosity(method="neutron_density", matrix="sandstone")])
    run = wf.run(well)
    for curve in ("PHID", "PHIN", "PHIT"):
        assert same(run.well.get(curve).data, reference.well.get(curve).data), curve


def test_saturation_matches_pp_interpret(well):
    reference = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model="archie", a=1, m=2, n=2)
    interpreted = copy.deepcopy(reference.well)
    interpreted.remove_curve("SW")
    interpreted.remove_curve("SXO")
    run = Workflow("x", [steps.saturation(model="archie", phie_curve="PHIE_CC")]).run(interpreted)
    assert same(run.well.get("SW").data, reference.well.get("SW").data)
    assert same(run.well.get("SXO").data, reference.well.get("SXO").data)


def test_net_pay_permeability_rock_quality_and_bvw_match_the_modules(reference):
    from petrophys.evaluation import NetPayModule, RockQualityModule, calculate_bvw
    from petrophys.interpretation import PermeabilityModule

    base = copy.deepcopy(reference.well)
    for curve in ("NET_ROCK", "NET_RES", "NET_PAY", "PERM", "RQI", "FZI", "PHI_Z", "BVW"):
        base.remove_curve(curve)
    wf = Workflow("x", [
        steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.1, sw_cutoff=0.6),
        steps.permeability(method="timur", swirr=0.2),
        steps.rock_quality(),
        steps.bvw(),
    ])
    run = wf.run(base)
    data = run.well
    vcl, phie, sw = (base.get(name).data for name in ("VCL", "PHIE", "SW"))
    pay = NetPayModule(0.35, 0.1, 0.6, "clastic").calculate(base, vcl=vcl, phie=phie, sw=sw)
    assert same(data.get("NET_PAY").data, pay.net_pay)
    assert same(data.get("NET_RES").data, pay.net_reservoir)
    perm = PermeabilityModule(method="timur").calculate(base, swi=0.2, phi=phie).perm
    assert same(data.get("PERM").data, perm)
    quality = RockQualityModule().calculate(base, perm=perm, phi_e=phie)
    assert same(data.get("FZI").data, quality.fzi) and same(data.get("RQI").data, quality.rqi)
    assert same(data.get("BVW").data, calculate_bvw(phie, sw))


def test_mud_filtrate_enables_invaded_zone_sxo_and_mhi(well):
    wf = Workflow("mhi", [
        steps.temperature(), steps.clay_volume(), steps.porosity(),
        steps.formation_water(rw=0.07, rw_temp_f=230),
        steps.mud_filtrate(rmf=0.3, rmf_temp_f=75),
        steps.saturation(), steps.mhi(),
    ])
    run = wf.run(well)
    assert run.steps[-2].parameters["sxo_method"].startswith("archie_rxo_rmf")
    mhi = run.well.get("MHI").data
    assert np.isfinite(mhi).any()
    assert same(mhi[np.isfinite(mhi)],
                (run.well.get("SW").data / run.well.get("SXO").data)[np.isfinite(mhi)])
    without = Workflow("x", [steps.clay_volume(), steps.porosity(),
                             steps.formation_water(rw=0.07, rw_temp_f=230), steps.saturation(),
                             steps.mhi()])
    missing = without.check(well)
    assert missing.loc[~missing["Met"], "Requires"].tolist() == ["RMF"]


def test_spectral_gamma_matches_the_function():
    from petrophys.interpretation.geochemistry import interpret_spectral_gamma

    source = _read("25_5-1")
    wf = Workflow("sgr", [steps.spectral_gamma()])
    table = wf.check(source)
    assert table["Met"].all() and table["Source"].iloc[0] == "well (THOR, URAN)"
    run = wf.run(source)
    expected = copy.deepcopy(source)
    result = interpret_spectral_gamma(expected)
    assert run.steps[0].output_curves == result.output_curves
    for curve in result.output_curves:
        assert same(run.well.get(curve).data, expected.get(curve).data)
    missing = wf.check(_read("HW-3"))
    assert not missing["Met"].iloc[0]


def test_standardize_adds_aliases_and_plans_them():
    costa = _read("HW-3")
    wf = Workflow("x", [steps.standardize(), steps.saturation(model="archie")])
    table = wf.check(costa)
    rt = table[table["Requires"] == "RDEP or RT"].iloc[0]
    assert rt["Met"] and rt["Source"] in ("step standardize", "well (RT)")
    run = Workflow("x", [steps.standardize()]).run(costa)
    assert "RDEP" in run.steps[0].output_curves
    assert run.well.get("RDEP") is not None and run.well.get("RT") is not None
