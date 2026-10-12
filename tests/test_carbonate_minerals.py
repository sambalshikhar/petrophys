"""Carbonate-domain mineral solve: calcite and dolomite from bulk density and closure."""
import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.lithology.carbonate_minerals import (
    solve_calcite_dolomite,
    solve_carbonate_minerals,
)
from petrophys.interpretation.lithology.costa_carbonate import solve_costa_carbonate
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.schema import describe_setting


def test_density_and_closure_recover_the_volumes():
    # 20% water, 10% clay (2.45), 50% calcite (2.71), 20% dolomite (2.87):
    # RHOB = 0.2*1.0 + 0.1*2.45 + 0.5*2.71 + 0.2*2.87 = 2.374 g/cm3.
    out = solve_calcite_dolomite(2.374, 0.2, 0.1, rho_fluid=1.0, rho_clay=2.45)

    assert out["solved"]
    assert out["calcite"] == pytest.approx(0.5, abs=1e-12)
    assert out["dolomite"] == pytest.approx(0.2, abs=1e-12)
    assert out["density_misfit"] == pytest.approx(0.0, abs=1e-12)


def test_fluid_density_sets_the_dolomite_share():
    # RHOB 2.45 g/cm3, porosity 0.20, no clay, COSTA dolomite 2.847 g/cm3. The
    # apparent matrix density (RHOB - 0.2*rho_fl) / 0.8 is 2.765 with a 1.19
    # brine and 2.805 with 1.03, so the dolomite share of the matrix,
    # (rho_maa - 2.71) / (2.847 - 2.71), is 0.4015 and 0.6934.
    def share(rho_fluid):
        out = solve_calcite_dolomite(
            2.45, 0.2, 0.0, rho_fluid=rho_fluid, rho_clay=2.52, rho_dolomite=2.847
        )
        return float(out["dolomite"]) / 0.8

    assert share(1.19) == pytest.approx(0.055 / 0.137, abs=1e-9)
    assert share(1.03) == pytest.approx(0.095 / 0.137, abs=1e-9)


def test_samples_outside_calcite_dolomite_are_left_unsolved():
    # Porosity 0.2 and no clay, so RHOB = 0.2 + 0.8*rho_maa. Within 0.02 g/cm3
    # of the calcite end point the volumes are clipped to pure calcite; quartz
    # (2.65, misfit 0.048) and anhydrite (2.96, misfit 0.072) are not solved.
    rho_maa = np.array([2.69, 2.65, 2.96, np.nan])
    out = solve_calcite_dolomite(0.2 + 0.8 * rho_maa, 0.2, 0.0, rho_fluid=1.0, rho_clay=2.45)

    np.testing.assert_array_equal(out["solved"], [True, False, False, False])
    assert out["calcite"][0] == pytest.approx(0.8) and out["dolomite"][0] == 0.0
    np.testing.assert_allclose(out["density_misfit"][:3], [0.016, 0.048, -0.072], atol=1e-12)
    assert np.isnan(out["calcite"][1:]).all() and np.isnan(out["dolomite"][1:]).all()


def test_density_derived_porosity_gives_its_matrix_back():
    # PHID = (2.87 - RHOB) / (2.87 - 1.0) assumes a dolomite matrix. With the
    # same fluid and no clay, rho_maa = (RHOB - PHID) / (1 - PHID) = 2.87 for
    # any RHOB, so the split returns pure dolomite whatever the rock is.
    rhob = np.array([2.30, 2.40, 2.50])
    phid = (2.87 - rhob) / (2.87 - 1.0)
    out = solve_calcite_dolomite(rhob, phid, 0.0, rho_fluid=1.0, rho_clay=2.45)

    np.testing.assert_allclose(out["dolomite"], 1.0 - phid, atol=1e-12)
    np.testing.assert_allclose(out["calcite"], 0.0, atol=1e-12)


def _costa_well(n_samples: int = 40) -> Well:
    rng = np.random.default_rng(7)
    well = Well("COSTA-LIKE")
    well.depth = np.arange(n_samples, dtype=float)
    well.depth_unit = "ft"
    phit = rng.uniform(0.1, 0.3, n_samples)
    gr = rng.uniform(10.0, 40.0, n_samples)
    illite = -0.0424035 + 0.00562239 * gr
    dolomite_share = rng.uniform(-0.6, 1.6, n_samples)  # some samples outside the end points
    matrix = 1.0 - phit - illite
    rhob = 1.19 * phit + 2.52 * illite + matrix * (2.71 + dolomite_share * (2.847 - 2.71))
    well.add_curve("GR", gr, "API", "Gamma Ray")
    well.add_curve("RHOB", rhob, "g/cm3", "Bulk Density")
    well.add_curve("PHIT", phit, "v/v", "Total Porosity")
    return well


def test_costa_calibration_reproduces_the_notebook_solve():
    well = _costa_well()
    notebook = solve_costa_carbonate(well)
    vectorised = solve_carbonate_minerals(well, "costa")

    assert notebook.solved.any() and not vectorised.solved.all()
    # The vectorised solve also accepts samples within 0.02 g/cm3 of an end point.
    assert np.all(vectorised.solved[notebook.solved])
    both = notebook.solved & vectorised.solved
    np.testing.assert_allclose(vectorised.v_calcite[both], notebook.v_calcite[both], atol=1e-5)
    np.testing.assert_allclose(vectorised.v_dolomite[both], notebook.v_dolomite[both], atol=1e-5)
    np.testing.assert_allclose(vectorised.v_clay, notebook.v_clay)
    assert vectorised.mineral_selection["fluid_density"] == 1.19


def test_generic_calibration_takes_clay_from_vcl():
    well = _costa_well()
    well.add_curve("VCL", np.full(well.n_samples, 0.1), "v/v", "Clay Volume")
    well.add_curve("PHIE", well.get("PHIT").data - 0.02, "v/v", "Effective Porosity")
    result = solve_carbonate_minerals(well, rho_fluid=1.05, rho_clay=2.5)
    expected = solve_calcite_dolomite(
        well.get("RHOB").data, well.get("PHIE").data, 0.1, rho_fluid=1.05, rho_clay=2.5
    )

    np.testing.assert_array_equal(result.v_dolomite, expected["dolomite"])
    np.testing.assert_array_equal(result.solved, expected["solved"])
    assert result.mineral_selection["porosity_curve"] == "PHIE"
    with pytest.raises(ValueError, match="calibration"):
        solve_carbonate_minerals(well, "field")


def _carbonate_well(n_samples: int = 40, name: str = "CARBONATE-TEST") -> Well:
    well = Well(name)
    well.depth = np.arange(8000.0, 8000.0 + 0.5 * n_samples, 0.5)
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(15.0, 60.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.35, 2.70, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.25, 0.05, n_samples), "v/v", "Neutron")
    well.add_curve("RDEP", np.linspace(50.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    return well


def _run(well_name: str = "CARBONATE-TEST", **settings):
    config = DynamicWorkflowConfig(
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
        lithology_domain="carbonate",
        **settings,
    )
    result = DynamicWorkflow(config).run(_carbonate_well(name=well_name))
    step = next(step for step in result.steps_executed if step.step == "lithology")
    return result, step


def test_current_profile_uses_vcl_rho_fl_and_rho_cl():
    result, step = _run(rho_fl=1.1, rho_cl=2.5)
    well = result.well
    expected = solve_calcite_dolomite(
        well.get("RHOB").data, well.get("PHIE").data, well.get("VCL").data,
        rho_fluid=1.1, rho_clay=2.5,
    )

    assert step.method == "generic_calcite_dolomite_density_closure"
    assert step.parameters["fluid_density"] == 1.1 and step.parameters["clay_density"] == 2.5
    assert step.input_curves == ["RHOB", "VCL", "PHIE"]
    assert not any(decision.startswith("PEF screened") for decision in step.decisions)
    np.testing.assert_array_equal(well.get("VDOLO").data, expected["dolomite"])
    assert well.get("VILL") is None


def test_trace_names_the_porosity_matrix_and_the_clay_curve():
    _, step = _run(matrix_type="dolomite")
    assert (
        "PHIE was computed in the porosity step with rho_ma 2.87 g/cm3, so the "
        "calcite-dolomite split is not independent of that matrix choice"
    ) in step.decisions

    well = _carbonate_well()
    well.add_curve("VSH", np.full(well.n_samples, 0.1), "v/v", "Shale Volume")
    config = DynamicWorkflowConfig(
        rw=0.05, skip_steps=["temperature", "rmf", "invasion", "mhi", "vclay"],
        lithology_domain="carbonate",
    )
    result = DynamicWorkflow(config).run(well)
    step = next(step for step in result.steps_executed if step.step == "lithology")
    assert step.input_curves[:2] == ["RHOB", "VSH"]
    assert step.parameters["clay_source"] == "VSH"
    assert any("clay = VSH at 2.45 g/cm3" in decision for decision in step.decisions)


def test_costa_calibration_is_an_explicit_option():
    result, step = _run(carbonate_calibration="COSTA")

    assert result.config.carbonate_calibration == "costa"
    assert step.method == "costa_calcite_dolomite_density_closure"
    assert step.parameters["fluid_density"] == 1.19
    assert "VILL" in step.output_curves


# A well of the eval package's main run, and 15/9-17, which its re-run interpreted.
@pytest.mark.parametrize("well_name", ["CARBONATE-TEST", "15/9-17"])
def test_benchmark_keeps_the_notebook_costa_route(well_name):
    result, step = _run(well_name, calculation_profile="benchmark")
    notebook = solve_costa_carbonate(result.well)

    assert step.method == "costa_gr_illite_calcite_dolomite"
    assert set(step.parameters) == {
        "illite_gr_intercept", "illite_gr_slope", "illite_equation",
        "lithology_curve_written", "lithology_source_priority",
        "multi_mineral_solved_samples", "reduced_quartz_calcite_samples",
        "multi_mineral_qc_flagged_samples", "multi_mineral_total_samples", "lithology_domain",
    }
    assert step.input_curves == ["RHOB", "VCL", "NPHI", "PHIE"]
    assert step.decisions == [
        "No GeoLink or CSV lithology was available -> inferred LITHO from mineral volumes",
        "PEF screened for plausibility; DTC used when PEF is absent or invalid",
        "Carbonate domain selected the COSTA GR-illite plus calcite/dolomite inversion",
        f"Multi-mineral solved {int(notebook.solved.sum())}/{result.well.n_samples} samples; "
        f"QC flagged {int(notebook.qc_flag.sum())}",
    ]
    np.testing.assert_array_equal(result.well.get("VCALC").data, notebook.v_calcite)
    np.testing.assert_array_equal(result.well.get("VILL").data, notebook.v_clay)


def test_generic_route_needs_vcl_and_costa_needs_gr():
    for profile, runs in (("current", False), ("benchmark", True)):
        config = DynamicWorkflowConfig(
            rw=0.05, skip_steps=["vclay"], lithology_domain="carbonate", calculation_profile=profile,
        )
        result = DynamicWorkflow(config).run(_carbonate_well())
        assert ("lithology" in result.steps_skipped) is not runs
        if not runs:
            assert "clay volume (VCL)" in result.specific_steps_skipped["lithology"]


def test_carbonate_calibration_setting_is_validated():
    assert DynamicWorkflowConfig().carbonate_calibration is None
    assert DynamicWorkflowConfig(carbonate_calibration=" Generic ").carbonate_calibration == "generic"
    with pytest.raises(ValueError, match="carbonate_calibration"):
        DynamicWorkflowConfig(carbonate_calibration="field")
    assert describe_setting("carbonate_calibration")["choices"] == ["generic", "costa"]
