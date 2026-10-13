"""Bad hole without an assumed bit size under "current"; the benchmark keeps 8.5 in.

Under ``calculation_profile="current"`` the caliper washout test
(CALI - BS > 1.5 in) and the large-hole test (BS > 17.5 in) run only where a
bit size is known, sample by sample (the BS curve, else a 3-36 in header bit
size), and |DRHO| > 0.15 g/cm3 flags washout wherever DRHO is logged. The
"geolink" profile keeps the eval package's rule: the BS curve as logged, else
the header bit size whatever its value, else an assumed 8.5 in bit, with
|DRHO| tested only on a well with a caliper.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from petrophys import Well
from petrophys.evaluation.diagnostics import HydrocarbonAnalysis
from petrophys.evaluation.diagnostics.numerical_statistical_analysis import NumericalStatisticalAnalysis
from petrophys.preprocessing.quality.bad_hole.flag_generator import (
    _known_bit_size,
    _plausible_bit_size,
    generate_bad_hole_flags,
)
from petrophys.reporting.reference_export import _build_reference_bad_hole_qc, _compact_curve_note
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, WorkflowStep

N = 8
#: Two samples of poor density-pad contact.
DRHO = [0.02, 0.02, 0.25, 0.02, 0.02, 0.25, 0.02, 0.02]
#: Benchmark trace parameters of the bad-hole step, which must not change.
BENCHMARK_PARAMETERS = {"bit_size_used", "bit_size_source", "bit_size_unique_values", "bit_size_depth_dependent"}


def _well(cali=12.5, drho=DRHO, bs=None, header=None) -> Well:
    """An in-gauge 12.25 in hole (CALI 12.5 in) at 2000.0-2003.5 m."""
    well = Well("BAD-HOLE")
    well.depth = 2000.0 + 0.5 * np.arange(N)
    well.depth_unit = "m"
    if cali is not None:
        well.add_curve("CALI", np.broadcast_to(np.asarray(cali, dtype=float), (N,)).copy(), "in", "Caliper")
    if drho is not None:
        well.add_curve("DRHO", np.asarray(drho, dtype=float), "g/cm3", "Density correction")
    if bs is not None:
        well.add_curve("BS", np.asarray(bs, dtype=float), "in", "Bit size")
    well.header.bit_size = header
    return well


def _bad_hole(well: Well, profile: str):
    config = DynamicWorkflowConfig(calculation_profile=profile)
    result = DynamicWorkflow(config).run_selected_steps(well, [WorkflowStep.BAD_HOLE])
    step = next(step for step in result.steps_executed if step.step == "bad_hole")
    curve = well.get("BAD_HOLE")
    flags = None if curve is None else curve.data.astype(int).tolist()
    return flags, step, result


@pytest.mark.parametrize("value, expected", [
    (8.5, 8.5), (3.0, 3.0), (36.0, 36.0), (2.9, None), (-999.25, None), (311.15, None),
    ("12.25", 12.25), ("n/a", None), (None, None),
])
def test_a_plausible_bit_size_is_3_to_36_in(value, expected):
    assert _plausible_bit_size(value) == expected


@pytest.mark.parametrize("bs, header, expected", [
    # The BS curve where it has a value, the header bit size in its gaps.
    ([8.5, np.nan, np.nan, 12.25], 12.25, [8.5, 12.25, 12.25, 12.25]),
    # A null or millimetre header fills nothing.
    ([8.5, np.nan, np.nan, 12.25], -999.25, [8.5, np.nan, np.nan, 12.25]),
    (None, 311.15, [np.nan] * 4),
    (None, 8.5, [8.5] * 4),
    # Nothing is assumed.
    (None, None, [np.nan] * 4),
])
def test_known_bit_size_is_the_bs_curve_else_a_plausible_header(bs, header, expected):
    well = Well("BIT-SIZE")
    well.depth = np.arange(4.0)
    if bs is not None:
        well.add_curve("BS", np.asarray(bs, dtype=float), "in", "Bit size")
    well.header.bit_size = header
    np.testing.assert_array_equal(_known_bit_size(well), np.asarray(expected, dtype=float))


def test_drho_alone_flags_washout_without_a_caliper_or_a_bit_size():
    no_caliper = _well(cali=None, bs=[8.5] * N)
    # By default |DRHO| is tested only on a well with a caliper and a bit size.
    assert not generate_bad_hole_flags(no_caliper)["washout"].any()
    assert generate_bad_hole_flags(no_caliper, drho_alone=True)["washout"].astype(int).tolist() == [
        0, 0, 1, 0, 0, 1, 0, 0
    ]
    no_bit_size = _well()
    assert not generate_bad_hole_flags(no_bit_size)["bad_hole"].any()
    assert generate_bad_hole_flags(no_bit_size, drho_alone=True)["bad_hole"].astype(int).tolist() == [
        0, 0, 1, 0, 0, 1, 0, 0
    ]


def test_without_a_bit_size_current_flags_drho_alone_and_the_benchmark_assumes_8_5_in():
    # CALI 12.5 in: in gauge for a 12.25 in bit, a 4 in washout for 8.5 in.
    flags, step, result = _bad_hole(_well(), "current")
    assert flags == [0, 0, 1, 0, 0, 1, 0, 0]
    assert step.parameters == {
        "bit_size_used": None,
        "bit_size_source": "none",
        "bit_size_unique_values": [],
        "bit_size_depth_dependent": False,
        "no_bit_size_samples": 8,
        "caliper_test_samples": 0,
        "not_assessed_samples": 0,
    }
    message = (
        "No bit size (BS curve or header) at 8 samples with a caliper reading (2000.0-2003.5 m): "
        "the caliper washout test (CALI - BS > 1.5 in) and the large-hole test (BS > 17.5 in) do "
        "not run there, and BAD_HOLE there flags only |DRHO| > 0.15 g/cm3."
    )
    assert step.decisions == [message, "2 bad-hole samples flagged out of 8"]
    assert result.warnings == [message]

    flags, step, result = _bad_hole(_well(), "geolink")
    assert flags == [1] * N
    assert step.parameters == {
        "bit_size_used": 8.5,
        "bit_size_source": "fallback_8.5",
        "bit_size_unique_values": [8.5],
        "bit_size_depth_dependent": False,
    }
    assert step.decisions == ["8 bad-hole samples flagged out of 8"]
    assert not result.warnings


def test_a_plausible_header_bit_size_fills_the_gaps_of_the_bs_curve_under_current_only():
    # BS logged over the upper half only (12.25 in); the header says 8.5 in.
    # "current" tests the lower half against 8.5 in: CALI - 8.5 = 4 in > 1.5 in.
    # The benchmark leaves the gap without a bit size, so only |DRHO| flags it.
    well = lambda: _well(bs=[12.25] * 4 + [np.nan] * 4, header=8.5, drho=[0.02, 0.02, 0.25] + [0.02] * 5)

    flags, step, result = _bad_hole(well(), "current")
    assert flags == [0, 0, 1, 0, 1, 1, 1, 1]
    params = step.parameters
    assert (params["bit_size_source"], params["bit_size_unique_values"], params["bit_size_depth_dependent"]) == (
        "BS_curve_and_well_header", [8.5, 12.25], True
    )
    assert (params["no_bit_size_samples"], params["caliper_test_samples"]) == (0, 8)
    assert not result.warnings

    flags, step, _ = _bad_hole(well(), "geolink")
    assert flags == [0, 0, 1, 0, 0, 0, 0, 0]
    assert set(step.parameters) == BENCHMARK_PARAMETERS
    assert (step.parameters["bit_size_source"], step.parameters["bit_size_used"]) == (
        "BS_curve_depth_by_depth", 12.25
    )


def test_where_the_bs_curve_is_null_current_says_so():
    # BS over the upper half; no header. CALI 14.0 in at samples 1 and 5:
    # sample 1 is 1.75 in over its 12.25 in bit, sample 5 has no bit size.
    # DRHO is logged over samples 0-1 only, so samples 4-7 meet neither test.
    cali = [12.5, 14.0, 12.5, 12.5, 12.5, 14.0, 12.5, 12.5]
    drho = [0.02, 0.02] + [np.nan] * 6
    flags, step, result = _bad_hole(_well(cali=cali, drho=drho, bs=[12.25] * 4 + [np.nan] * 4), "current")

    assert flags == [0, 1, 0, 0, 0, 0, 0, 0]
    params = step.parameters
    assert (params["bit_size_source"], params["bit_size_used"]) == ("BS_curve_depth_by_depth", 12.25)
    assert (params["no_bit_size_samples"], params["caliper_test_samples"], params["not_assessed_samples"]) == (
        4, 4, 4
    )
    assert result.warnings == [
        "No bit size (BS curve or header) at 4 samples with a caliper reading (2002.0-2003.5 m): "
        "the caliper washout test (CALI - BS > 1.5 in) and the large-hole test (BS > 17.5 in) do "
        "not run there, and BAD_HOLE there flags only |DRHO| > 0.15 g/cm3."
    ]
    assert step.decisions[1:] == [
        "Bad hole not assessed at 4 samples with neither a DRHO reading nor a caliper reading "
        "with a bit size; BAD_HOLE is 0 there",
        "1 bad-hole samples flagged out of 8",
    ]


def test_without_a_caliper_current_flags_drho_and_the_benchmark_does_not():
    flags, step, result = _bad_hole(_well(cali=None, bs=[8.5] * N), "current")
    assert flags == [0, 0, 1, 0, 0, 1, 0, 0]
    assert step.decisions == [
        "No caliper reading: washout is |DRHO| > 0.15 g/cm3 alone",
        "2 bad-hole samples flagged out of 8",
    ]
    assert not result.warnings

    flags, _, _ = _bad_hole(_well(cali=None, bs=[8.5] * N), "geolink")
    assert flags == [0] * N


def test_a_caliper_without_a_bit_size_or_drho_is_not_assessed_under_current():
    reason = (
        "Needs a density-correction (DRHO) curve, or a caliper (CALI) and a bit size "
        "(BS curve or a 3-36 in header bit size)"
    )
    well = _well(drho=None)
    well.add_curve("GR", np.linspace(30.0, 120.0, N), "API", "Gamma Ray")
    result = DynamicWorkflow(DynamicWorkflowConfig(rw=0.05)).run(well)
    assert result.specific_steps_skipped["bad_hole"] == reason
    assert well.get("BAD_HOLE") is None

    # Run on its own, the step writes no flag and says why.
    flags, step, result = _bad_hole(_well(drho=None), "current")
    assert flags is None and step.output_curves == []
    assert step.decisions == [
        "Bad hole not assessed: no DRHO reading and no caliper reading with a bit size "
        "(BS curve or header), so BAD_HOLE is not written"
    ]
    assert not result.warnings

    # The benchmark tests the caliper against 8.5 in: CALI - 8.5 = 4 in > 1.5 in.
    well = _well(drho=None)
    well.add_curve("GR", np.linspace(30.0, 120.0, N), "API", "Gamma Ray")
    result = DynamicWorkflow(DynamicWorkflowConfig(rw=0.05, calculation_profile="geolink")).run(well)
    assert "bad_hole" not in result.specific_steps_skipped
    assert well.get("BAD_HOLE").data.tolist() == [1.0] * N


def test_the_clay_point_is_picked_from_in_gauge_shale_without_a_bit_size():
    # 40 shale samples in gauge in a 12.25 in hole (CALI 12.5 in, DRHO 0.02),
    # half at (2.33, 0.38) and half at (2.37, 0.42): median (2.35, 0.40). Six
    # washed-out shale samples read (2.00, 0.55) with DRHO 0.25 g/cm3. No bit
    # size: "current" flags only the six, and picks the point from the 40.
    rhob = np.r_[np.full(20, 2.33), np.full(20, 2.37), np.full(6, 2.00), np.full(30, 2.2375), np.full(4, 2.40)]
    nphi = np.r_[np.full(20, 0.38), np.full(20, 0.42), np.full(6, 0.55), np.full(30, 0.23), np.full(4, 0.25)]
    gr = np.r_[np.full(46, 120.0), np.full(30, 30.0), np.full(4, 80.0)]
    drho = np.r_[np.full(40, 0.02), np.full(6, 0.25), np.full(34, 0.02)]
    codes = np.r_[np.full(46, 8.0), np.full(34, 1.0)]
    well = Well("ND-CLAY-POINT")
    well.depth = 2000.0 + 0.5 * np.arange(rhob.size)
    well.depth_unit = "m"
    for name, values, unit in (("GR", gr, "API"), ("RHOB", rhob, "g/cc"), ("NPHI", nphi, "v/v"),
                               ("CALI", np.full(rhob.size, 12.5), "in"), ("DRHO", drho, "g/cm3"),
                               ("LITHOLOGY_GEOLINK", codes, "")):
        well.add_curve(name, values, unit, name)

    result = DynamicWorkflow(DynamicWorkflowConfig(rw=0.05, gr_min=20.0, gr_max=120.0)).run(well)
    params = next(step for step in result.steps_executed if step.step == "vclay").parameters

    assert int(np.nansum(well.get("BAD_HOLE").data)) == 6
    assert (params["nd_clay_source"], params["nd_clay_basis"], params["nd_clay_samples"]) == (
        "shale_samples", "lithology_shale", 40
    )
    assert params["nd_clay_density"] == pytest.approx(2.35)
    assert params["nd_clay_neutron"] == pytest.approx(0.40)


@pytest.mark.parametrize("profile, bad_hole, crossover, poor", [
    # DRHO alone flags samples 2 and 5; elsewhere the gas crossover counts.
    ("current", [0, 0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0]),
    # CALI - 8.5 in = 4 in flags every sample: no crossover, all poor quality.
    ("geolink", [1] * N, [0] * N, [1] * N),
])
def test_crossover_and_log_quality_follow_the_bad_hole_flag(profile, bad_hole, crossover, poor):
    # Gas crossover (PHID - PHIN = 0.15 > 0.05) in clean, porous rock.
    well = _well()
    for name, value in (("PHID", 0.30), ("PHIN", 0.15), ("VCL", 0.10), ("PHIE", 0.20), ("SW", 0.30)):
        well.add_curve(name, np.full(N, value), "v/v", name)
    DynamicWorkflow(DynamicWorkflowConfig(calculation_profile=profile)).run_selected_steps(
        well, [WorkflowStep.BAD_HOLE]
    )
    HydrocarbonAnalysis().analyze(well, add_curves=True)
    quality = NumericalStatisticalAnalysis().analyze(well, add_curves=False).sections[
        "log_quality_control_assessment"
    ].raw_labels

    assert well.get("BAD_HOLE").data.astype(int).tolist() == bad_hole
    assert well.get("CROSSOVER_FLAG").data.astype(int).tolist() == crossover
    assert [int(label == "POOR_LOG_QUALITY") for label in quality] == poor


def _workflow_result(profile: str, bit_size_used=None, source="none"):
    """The parts of a workflow result the context exporter reads."""
    step = SimpleNamespace(
        step="bad_hole", success=True, method="caliper_drho_flags", input_curves=["CALI", "DRHO"],
        output_curves=["BAD_HOLE"], decisions=[],
        parameters={"bit_size_used": bit_size_used, "bit_size_source": source},
    )
    run = SimpleNamespace(
        config=SimpleNamespace(calculation_profile=profile), steps_executed=[step], steps_skipped=[],
        warnings=[], errors=[],
    )
    return SimpleNamespace(base_result=run, zone_results=[]), step


def test_the_context_bad_hole_check_assumes_no_bit_size_under_current():
    well = _well()
    well.add_curve("BAD_HOLE", np.asarray([0, 0, 1, 0, 0, 1, 0, 0], dtype=float), "", "Bad Hole Flag")
    workflow_result, step = _workflow_result("current")

    qc = _build_reference_bad_hole_qc(well, workflow_result)
    assert (qc["bad_hole_points"], qc["bad_hole_sequences"]) == (2, ["2001.0-2001.0 m", "2002.5-2002.5 m"])
    assert (qc["no_bit_size_points"], qc["not_assessed_points"]) == (8, 0)
    assert qc["log_availability"]["BS"] is False
    assert qc["parameters_used"]["bit_size_used_in"] is None
    assert qc["parameters_used"]["washout_formula"] == (
        "(CALI - BS) > 1.50 in OR |DRHO| > 0.15 g/cc; |DRHO| > 0.15 g/cc alone where no bit size is known"
    )
    assert qc["summary_text"].endswith(
        "final BAD_HOLE=2 of 8 samples (25.00%). No bit size (BS curve or header) for 8 samples: "
        "washout there is |DRHO| > 0.15 g/cc alone."
    )
    assert _compact_curve_note("BAD_HOLE", "computed", "", step, well, benchmark=False) == (
        "BAD_HOLE: washout=(CALI-BS)>1.50 in or |DRHO|>0.15 g/cc; large_hole=BS>17.50 in; "
        "rule=washout OR large_hole; no bit size (BS curve or header), so washout=|DRHO|>0.15 g/cc "
        "alone; flagged=2/8 (25%)."
    )

    # A caliper with neither a bit size nor DRHO cannot be screened.
    qc = _build_reference_bad_hole_qc(_well(drho=None), workflow_result)
    assert qc["summary_text"] == (
        "Bad-hole screening could not be evaluated: no DRHO reading and no caliper reading with a "
        "bit size (BS curve or header)."
    )
    assert (qc["bad_hole_points"], qc["no_bit_size_points"], qc["not_assessed_points"]) == (0, 8, 8)


def test_the_context_bad_hole_check_keeps_8_5_in_under_the_benchmark():
    well = _well()
    well.add_curve("BAD_HOLE", np.ones(N), "", "Bad Hole Flag")
    workflow_result, step = _workflow_result("geolink", bit_size_used=8.5, source="fallback_8.5")

    qc = _build_reference_bad_hole_qc(well, workflow_result)
    assert (qc["bad_hole_points"], qc["parameters_used"]["bit_size_used_in"]) == (8, 8.5)
    assert qc["parameters_used"]["washout_formula"] == "(CALI - BS) > 1.50 in OR |DRHO| > 0.15 g/cc"
    assert qc["summary_text"] == (
        "Bad-hole screening uses washout OR large-hole logic: washout=8, large_hole=0, "
        "final BAD_HOLE=8 of 8 samples (100.00%)."
    )
    assert not {"no_bit_size_points", "not_assessed_points"} & set(qc)
    assert _compact_curve_note("BAD_HOLE", "computed", "", step, well, benchmark=True) == (
        "BAD_HOLE: washout=(CALI-BS)>1.50 in or |DRHO|>0.15 g/cc; large_hole=BS>17.50 in; "
        "rule=washout OR large_hole; BS=8.5 in; flagged=8/8 (100%)."
    )


@pytest.mark.parametrize("profile, reason", [
    ("current", "Needs a density-correction (DRHO) curve, or a caliper (CALI) and a bit size "
                "(BS curve or a 3-36 in header bit size)"),
    ("geolink", "Needs a caliper (CALI) or density-correction (DRHO) curve"),
])
def test_without_caliper_or_drho_the_step_is_skipped(profile, reason):
    well = _well(cali=None, drho=None, bs=[8.5] * N)
    well.add_curve("GR", np.linspace(30.0, 120.0, N), "API", "Gamma Ray")
    result = DynamicWorkflow(DynamicWorkflowConfig(rw=0.05, calculation_profile=profile)).run(well)
    assert result.specific_steps_skipped["bad_hole"] == reason
