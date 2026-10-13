"""Mineral solver versions: version 1 (original closure-gated solver) and version 2 (current)."""
import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation import (
    MultiMineralSolver,
    MultiMineralSolverV1,
    MultiMineralSolverV2,
)
from petrophys.interpretation.lithology import LithologyModule
from petrophys.interpretation.lithology.multi_mineral_solver_v1 import ROW_UNCERTAINTY
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.schema import describe_setting

# Clean water-filled quartz with 20% porosity and no clay:
# RHOB = 0.8*2.65 + 0.2*1.0, NPHI = 0.8*-0.02 + 0.2*1.0, U = 0.8*4.8 + 0.2*0.36.
RHOB = 2.32
NPHI = 0.184
PEF = (0.8 * 4.8 + 0.2 * 0.36) / RHOB
# DT = 0.8*55.5 + 0.2*189.0 for the same rock.
DT = 82.2


def test_versions_are_labelled():
    assert MultiMineralSolverV1.version == 1
    assert MultiMineralSolver.version == 2
    assert MultiMineralSolverV2 is MultiMineralSolver


def test_v1_solves_a_consistent_sample():
    result = MultiMineralSolverV1().solve(RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF)

    assert result["quartz"][0] == pytest.approx(0.8, abs=1e-6)
    for mineral in ("calcite", "dolomite", "anhydrite"):
        assert result[mineral][0] == pytest.approx(0.0, abs=1e-6)
    assert result["solved"][0] and not result["qc_flag"][0]


def test_v1_leaves_samples_unsolved_when_closure_fails():
    # The logs say 80% matrix, but VCL + PHIE leave only 60%: the totals differ by 0.2 > 0.05.
    result = MultiMineralSolverV1().solve(RHOB, NPHI, vcl=0.0, phie=0.4, pef=PEF)

    assert np.isnan(result["quartz"][0])
    assert result["qc_flag"][0] and not result["solved"][0]


def test_v1_marks_missing_inputs_unsolved():
    result = MultiMineralSolverV1().solve([RHOB, np.nan], [NPHI, NPHI], vcl=[0.0, 0.0], phie=[0.2, 0.2])

    assert np.isfinite(result["quartz"][0])
    assert np.isnan(result["quartz"][1]) and result["qc_flag"][1]


def test_v1_ignores_core_porosity():
    plain = MultiMineralSolverV1().solve(RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF)
    with_core = MultiMineralSolverV1().solve(RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, core_phi=0.35)

    assert with_core["quartz"][0] == plain["quartz"][0]


def test_lithology_module_selects_the_solver_version():
    assert type(LithologyModule().solver) is MultiMineralSolver
    assert type(LithologyModule(solver_version=1).solver) is MultiMineralSolverV1
    with pytest.raises(ValueError, match="solver_version"):
        LithologyModule(solver_version=3)


def test_workflow_setting_defaults_to_version_1_and_is_validated():
    assert DynamicWorkflowConfig().mineral_solver_version == 1
    assert DynamicWorkflowConfig(mineral_solver_version=np.int64(2)).mineral_solver_version == 2
    for bad in (0, 3, "2", True, 1.0):
        with pytest.raises(ValueError, match="mineral_solver_version"):
            DynamicWorkflowConfig(mineral_solver_version=bad)
    assert describe_setting("mineral_solver_version")["choices"] == [1, 2]


def _workflow_well() -> Well:
    n_samples = 40
    well = Well("SOLVER-VERSION-TEST")
    well.depth = np.arange(8000.0, 8020.0, 0.5)
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(25.0, 125.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.32, n_samples), "v/v", "Neutron")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n_samples), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(50.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    return well


def _lithology_step(version: int):
    config = DynamicWorkflowConfig(
        standardize_input=False,
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
        mineral_solver_version=version,
    )
    result = DynamicWorkflow(config).run(_workflow_well())
    assert result.success
    return next(step for step in result.steps_executed if step.step == "lithology")


@pytest.mark.parametrize(
    "version, method",
    [(1, "canonical_multi_mineral_solver"), (2, "multi_mineral_solver")],
)
def test_workflow_runs_the_selected_solver(version, method):
    step = _lithology_step(version)

    assert step.method == method
    assert step.parameters["mineral_solver_version"] == version


def test_v1_weights_rows_by_tool_uncertainty():
    # The same quartz sample with the sonic reading 5 us/ft slow. Only quartz
    # is active, so the solve is a one-unknown least squares,
    # x = sum(w^2 a b) / sum(w^2 a^2), with rows a = (1, 2.65, -0.02, 4.8, 55.5)
    # whose right-hand sides fit x = 0.8 except DT (+5 us/ft):
    # unscaled (w = 1):  x = 0.8 + 55.5*5 / 3111.31 = 0.8892, closure miss 0.089 > 0.05;
    # weighted (w = 1/sigma = 100, 50, 50, 2, 0.5):
    #                    x = 0.8 + 0.25*55.5*5 / 28419.47 = 0.8024, closure miss 0.0024.
    weighted = MultiMineralSolverV1().solve(RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, dt=DT + 5.0)
    unscaled = MultiMineralSolverV1(row_uncertainty=None).solve(
        RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, dt=DT + 5.0
    )

    assert weighted["solved"][0]
    assert weighted["quartz"][0] == pytest.approx(0.8, abs=1e-6)
    for mineral in ("calcite", "dolomite", "anhydrite"):
        assert weighted[mineral][0] == pytest.approx(0.0, abs=1e-6)
    assert not unscaled["solved"][0] and np.isnan(unscaled["quartz"][0])


def test_v1_weighted_solve_matches_the_hand_computed_least_squares():
    # Quartz only, sonic 5 us/ft fast: the one-unknown weighted least squares
    # x = 0.8 + w_dt^2 * 55.5 * (-5) / sum(w^2 a^2) with sum(w^2 a^2) =
    # 100^2*1 + 50^2*2.65^2 + 50^2*0.02^2 + 2^2*4.8^2 + 0.5^2*55.5^2 = 28419.4725.
    # It stays below the 0.8 available, so it is reported as solved.
    quartz_only = {"minerals": ["quartz"], "minerals_no_pef": ["quartz"]}
    weighted = MultiMineralSolverV1(**quartz_only).solve(
        RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, dt=DT - 5.0
    )
    # Unscaled: x = 0.8 - 55.5*5 / 3111.3129 = 0.7108, a closure miss of 0.089.
    unscaled = MultiMineralSolverV1(row_uncertainty=None, **quartz_only).solve(
        RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, dt=DT - 5.0
    )

    assert weighted["quartz"][0] == pytest.approx(0.8 - 0.25 * 55.5 * 5.0 / 28419.4725, abs=1e-9)
    assert weighted["solved"][0]
    assert np.isnan(unscaled["quartz"][0]) and not unscaled["solved"][0]


def test_v1_weighted_and_unscaled_agree_on_consistent_logs():
    for solver in (MultiMineralSolverV1(), MultiMineralSolverV1(row_uncertainty=None)):
        result = solver.solve(RHOB, NPHI, vcl=0.0, phie=0.2, pef=PEF, dt=DT)
        assert result["quartz"][0] == pytest.approx(0.8, abs=1e-6)


def test_v1_row_uncertainty_defaults_and_validation():
    assert MultiMineralSolverV1().row_uncertainty == dict(ROW_UNCERTAINTY)
    assert dict(ROW_UNCERTAINTY) == {"unity": 0.01, "rhob": 0.02, "nphi": 0.02, "u": 0.5, "dt": 2.0}
    assert MultiMineralSolverV1(row_uncertainty=None).row_uncertainty is None
    merged = MultiMineralSolverV1(row_uncertainty={"rhob": 0.03}).row_uncertainty
    assert merged == {**ROW_UNCERTAINTY, "rhob": 0.03}
    with pytest.raises(ValueError, match="row_uncertainty keys"):
        MultiMineralSolverV1(row_uncertainty={"pef": 0.1})
    for bad in (0.0, -0.02, np.nan, True):
        with pytest.raises(ValueError, match="positive number"):
            MultiMineralSolverV1(row_uncertainty={"dt": bad})


def test_lithology_module_passes_solver_options():
    assert LithologyModule(solver_version=1).solver.row_uncertainty == dict(ROW_UNCERTAINTY)
    unscaled = LithologyModule(solver_version=1, solver_options={"row_uncertainty": None})
    assert unscaled.solver.row_uncertainty is None
    assert LithologyModule(solver_options={"closure_tolerance": 0.1}).solver.closure_tolerance == 0.1


@pytest.mark.parametrize("profile", ["current", "geolink"])
def test_workflow_weights_version_1_rows_only_under_current(profile):
    well = _workflow_well()
    well.add_curve("DTC", np.linspace(70.0, 115.0, well.n_samples), "us/ft", "Sonic")
    config = DynamicWorkflowConfig(
        standardize_input=False,
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
        calculation_profile=profile,
    )
    result = DynamicWorkflow(config).run(well)
    step = next(step for step in result.steps_executed if step.step == "lithology")
    weighted = profile == "current"

    expected = LithologyModule(
        lithology="auto",
        solver_version=1,
        solver_options=None if weighted else {"row_uncertainty": None},
    ).calculate(result.well)
    np.testing.assert_array_equal(result.well.get("VQUAR").data, expected.v_quartz)
    assert step.parameters["multi_mineral_solved_samples"] == int(np.count_nonzero(expected.solved))
    assert ("row_uncertainty" in step.parameters) is weighted
    assert any("tool uncertainty" in decision for decision in step.decisions) is weighted
