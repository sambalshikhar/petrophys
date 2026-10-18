import numpy as np
import pytest

from petrophys.model import Well
from petrophys.interpretation.core_analysis.calibration import (
    correct_porosity_with_grain_density,
    fit_poro_perm_transform,
    interpolate_log_at_core,
)
from petrophys.model.core import CoreData
from petrophys.interpretation.permeability.perm_module import PermeabilityModule
from petrophys.workflow.dynamic_workflow import DynamicWorkflow, DynamicWorkflowConfig, WorkflowStep


def _core_well(core_phi: np.ndarray, core_perm: np.ndarray) -> Well:
    well = Well("CORE-CORRECTION")
    well.depth = np.arange(1000.0, 1010.0)
    phie = np.linspace(0.10, 0.28, well.depth.size)
    phic = np.full(well.depth.size, np.nan)
    kh = np.full(well.depth.size, np.nan)
    core_idx = np.array([0, 2, 4, 6, 8])
    phic[core_idx] = core_phi
    kh[core_idx] = core_perm
    well.add_curve("PHIE", phie, "v/v", "Effective porosity")
    well.add_curve("SW", np.full(well.depth.size, 0.35), "v/v", "Water saturation")
    well.add_curve("RDEP", np.full(well.depth.size, 10.0), "ohm.m", "Deep resistivity")
    well.add_curve("RESFLAG", np.ones(well.depth.size), "", "Reservoir flag")
    well.add_curve("PHIC", phic, "v/v", "Core porosity")
    well.add_curve("KH_CORE", kh, "mD", "Core permeability")
    return well


def _cored_well(core_phi_from_phit, n_core: int = 10) -> Well:
    """PHIT 0.06-0.288 and PHIE = PHIT - 0.03 (0 in two shale samples), cored every other depth."""
    n = 2 * n_core
    well = Well("CORED")
    well.depth = 1000.0 + 0.5 * np.arange(n)
    well.depth_unit = "m"
    phit = 0.06 + 0.012 * np.arange(n)
    phie = phit - 0.03
    phie[:2] = 0.0
    core_idx = np.arange(0, n, 2)
    phic = np.full(n, np.nan)
    kh = np.full(n, np.nan)
    phic[core_idx] = core_phi_from_phit(phit[core_idx])
    kh[core_idx] = 10 ** (12.0 * phic[core_idx] - 2.0)
    well.add_curve("PHIT", phit, "v/v", "Total porosity")
    well.add_curve("PHIE", phie, "v/v", "Effective porosity")
    well.add_curve("VCL", (phit - phie) / 0.15, "v/v", "Clay volume")
    well.add_curve("SW", np.full(n, 0.35), "v/v", "Water saturation")
    well.add_curve("RDEP", np.full(n, 10.0), "ohm.m", "Deep resistivity")
    well.add_curve("RESFLAG", np.ones(n), "", "Reservoir flag")
    well.add_curve("PHIC", phic, "v/v", "Core porosity")
    well.add_curve("KH_CORE", kh, "mD", "Core permeability")
    return well


def _linear_core(phit):
    return 1.1 * phit - 0.02


def test_same_well_core_correction_is_used_when_core_pairs_are_available():
    phie = np.linspace(0.10, 0.28, 10)[[0, 2, 4, 6, 8]]
    core_phi = 1.1 * phie + 0.01
    core_perm = 10 ** (12.0 * core_phi - 2.0)
    well = _core_well(core_phi, core_perm)

    result = PermeabilityModule(method="auto").calculate(well, swi=0.25)

    assert result.metadata["same_well_core_models"] == 1
    assert result.metadata["peer_core_models"] == 0
    assert result.metadata["phi_corecal"] is not None
    assert result.metadata["perm_core"] is not None
    assert "porosity_r2" not in result.metadata["core_models"][0]
    assert "perm_r2" not in result.metadata["core_models"][0]
    assert "porosity_rmse" not in result.metadata["core_models"][0]
    assert "perm_rmse_log10" not in result.metadata["core_models"][0]
    assert result.metadata["core_models"][0]["core_porosity_curve"] == "PHIC"
    assert result.metadata["core_models"][0]["core_permeability_curve"] == "KH_CORE"
    assert np.count_nonzero(np.isfinite(result.metadata["perm_core"])) == well.n_samples


def test_same_well_core_correction_does_not_reject_on_fit_accuracy_metrics():
    core_phi = np.array([0.18, 0.11, 0.24, 0.13, 0.20])
    core_perm = np.array([10.0, 300.0, 2.0, 80.0, 6.0])
    well = _core_well(core_phi, core_perm)

    result = PermeabilityModule(method="auto").calculate(well, swi=0.25)

    assert result.metadata["same_well_core_models"] == 1
    assert result.metadata["peer_core_models"] == 0
    assert result.metadata["phi_corecal"] is not None
    assert result.metadata["perm_core"] is not None
    assert result.metadata["core_rejected_models"] == []
    assert np.count_nonzero(np.isfinite(result.perm)) == well.n_samples


def test_dynamic_workflow_rewrites_stale_core_curves_from_current_core_model():
    well = _cored_well(_linear_core)
    well.add_curve("PHIE_CC", np.full(well.n_samples, 0.30), "v/v", "Stale core porosity")
    well.add_curve("PERM_CC", np.full(well.n_samples, 5000.0), "mD", "Stale core perm")

    result = DynamicWorkflow().run_selected_steps(well, [WorkflowStep.PERMEABILITY])

    assert result.well.get("PERM") is not None
    assert result.well.get("PHIE_CC") is not None
    assert result.well.get("PERM_CC") is not None
    assert not np.allclose(result.well.get("PHIE_CC").data, 0.30)
    assert not np.allclose(result.well.get("PERM_CC").data, 5000.0)


def test_dynamic_workflow_builds_phie_cc_before_saturation():
    well = _cored_well(_linear_core)

    result = DynamicWorkflow(
        DynamicWorkflowConfig(
            rw=0.05,
            skip_steps=[
                "vclay",
                "rcl",
                "rmf",
                "hc_correction",
                "porosity",
                "lithology",
                "invasion",
                "permeability",
                "rock_quality",
                "bvw",
                "mhi",
                "net_pay",
            ],
        )
    ).run(well)

    executed = [step.step for step in result.steps_executed]
    assert executed.index("core_correction") < executed.index("saturation")
    assert result.well.get("PHIE_CC") is not None
    saturation_step = next(step for step in result.steps_executed if step.step == "saturation")
    assert "PHIE_CC" in saturation_step.input_curves


@pytest.mark.parametrize("profile", ["current", "geolink"])
@pytest.mark.parametrize("core_perm", ["absent", "all_null", "porosity_alias"])
def test_core_porosity_does_not_require_core_permeability_in_full_routing(profile, core_perm, tmp_path):
    from petrophys.io import export_las, read_las
    from petrophys.reporting.reference_export import _compact_curve_note

    config = DynamicWorkflowConfig(calculation_profile=profile, rw=0.05, swirr=0.25)
    steps = [WorkflowStep.RW, WorkflowStep.CORE_CORRECTION, WorkflowStep.SATURATION, WorkflowStep.PERMEABILITY]
    paired = DynamicWorkflow(config).run_selected_steps(_cored_well(_linear_core), steps)
    well = _cored_well(_linear_core)
    if core_perm in {"absent", "porosity_alias"}:
        well.remove_curve("KH_CORE")
        if core_perm == "porosity_alias":
            well.rename_curve("PHIC", "CPOR")
    else:
        well.get("KH_CORE").data[:] = np.nan
    result = DynamicWorkflow(config).run_selected_steps(well, steps)

    assert not result.errors
    assert result.well.get("PHIE_CC") is not None
    np.testing.assert_allclose(result.well["PHIE_CC"].data, paired.well["PHIE_CC"].data)
    np.testing.assert_allclose(result.well["SW"].data, paired.well["SW"].data)
    assert result.well.get("PERM_CC") is None
    assert result.well.get("PERM") is not None
    saturation = next(step for step in result.steps_executed if step.step == "saturation")
    assert ("PHIE_CC" if profile == "current" else "PHIE") in saturation.input_curves

    target = tmp_path / "processed.las"
    export_las(result.well, target)
    restored = read_las(target)
    np.testing.assert_allclose(restored["PHIE_CC"].data, result.well["PHIE_CC"].data, atol=1e-5)
    note = _compact_curve_note("PHIE_CC", "computed", "", _core_step(result), result.well)
    assert "core porosity correction" in note
    assert "porosity-only calibration" in note


def test_empirical_permeability_keeps_porosity_calibrated_earlier_in_the_run():
    well = _cored_well(_linear_core)
    well.remove_curve("KH_CORE")
    config = DynamicWorkflowConfig(perm_method="timur", swirr=0.25)
    result = DynamicWorkflow(config).run_selected_steps(
        well, [WorkflowStep.CORE_CORRECTION, WorkflowStep.PERMEABILITY]
    )
    assert not result.errors
    corrected_total = 1.1 * well["PHIT"].data - 0.02
    expected = np.clip(corrected_total - (well["PHIT"].data - well["PHIE"].data), 0, corrected_total)
    np.testing.assert_allclose(result.well["PHIE_CC"].data, expected)
    assert result.well.get("PERM_CC") is None


def test_porosity_only_formation_fits_are_not_overwritten_by_global_permeability_fallback():
    from petrophys.model.formation_tops import FormationTops

    well = Well("POROSITY-ONLY-FORMATIONS")
    well.depth = 1000.0 + np.arange(40) * 0.5
    phi = np.linspace(.1, .3, well.n_samples)
    expected = np.where(well.depth < 1010., .8 * phi + .02, 1.2 * phi - .015)
    well.add_curve("PHIE", phi, "v/v")
    well.add_curve("PHIC", expected, "v/v")
    well.add_curve("SW", np.full(well.n_samples, .4), "v/v")
    tops = FormationTops.from_dict([
        {"name": "Upper", "top_depth": 1000., "bottom_depth": 1009.75},
        {"name": "Lower", "top_depth": 1010., "bottom_depth": 1020.},
    ])
    result = PermeabilityModule().calculate(well, swi=.25, formation_tops=tops)
    np.testing.assert_allclose(result.metadata["phi_corecal"], expected, atol=1e-12)
    assert result.metadata["perm_core"] is None
    assert np.isfinite(result.perm).all()


@pytest.mark.parametrize("method", ["auto", "timur"])
def test_standard_workflow_corrects_porosity_without_core_permeability(method):
    from petrophys.workflow.standard_workflow import StandardWorkflow, WorkflowConfig

    well = _cored_well(_linear_core)
    well.remove_curve("KH_CORE")
    with pytest.warns(DeprecationWarning):
        workflow = StandardWorkflow(WorkflowConfig(
            calculate_vcl=False, calculate_porosity=False, calculate_net_pay=False,
            rw=.05, swirr=.25, perm_method=method,
        ))
    result = workflow.run(well)
    assert result.success
    assert result.well.get("PHIE_CC") is not None
    assert result.well.get("PERM_CC") is None
    saturation = next(step for step in result.steps if step.step == "saturation")
    assert "PHIE_CC" in saturation.input_curves


def test_peer_core_is_not_used_unless_explicitly_enabled():
    phie = np.linspace(0.10, 0.28, 10)[[0, 2, 4, 6, 8]]
    peer = _core_well(1.1 * phie + 0.01, 10 ** (12.0 * (1.1 * phie + 0.01) - 2.0))
    target = Well("TARGET-NO-CORE")
    target.depth = np.arange(1000.0, 1010.0)
    target.add_curve("PHIE", np.linspace(0.10, 0.28, target.depth.size), "v/v", "Effective porosity")

    result = PermeabilityModule(method="auto").calculate(target, swi=0.25, peer_wells=[peer])

    assert result.metadata["peer_wells_considered"] == 1
    assert result.metadata["peer_core_enabled"] is False
    assert result.metadata["peer_core_models"] == 0
    assert result.metadata["perm_core"] is None


def test_core_log_interpolation_sorts_descending_depth_and_averages_duplicates():
    well = Well("CORE-INTERP")
    well.depth = np.array([102.0, 100.0, 101.0, 101.0])
    well.add_curve("PHIE", np.array([0.30, 0.10, 0.20, 0.22]), "v/v", "Porosity")
    core = CoreData(depth=np.array([100.0, 101.0, 102.0]), porosity=np.ones(3))

    values = interpolate_log_at_core(well, core, "PHIE")

    np.testing.assert_allclose(values, [0.10, 0.21, 0.30])


def test_grain_density_correction_interpolates_descending_core_depth():
    well = Well("GDEN-DESC")
    well.depth = np.array([100.0, 101.0, 102.0])
    well.add_curve("RHOB", np.array([2.30, 2.30, 2.30]), "g/cc", "Bulk density")
    core = CoreData(
        depth=np.array([102.0, 100.0]),
        grain_density=np.array([2.80, 2.60]),
    )

    phid = correct_porosity_with_grain_density(well, core)

    np.testing.assert_allclose(phid, [(2.60 - 2.30) / 1.60, (2.70 - 2.30) / 1.70, (2.80 - 2.30) / 1.80])


def test_poro_perm_transform_requires_two_valid_core_pairs():
    core = CoreData(
        depth=np.array([100.0, 101.0]),
        porosity=np.array([0.20, np.nan]),
        permeability=np.array([10.0, 20.0]),
    )

    with pytest.raises(ValueError, match="At least two valid"):
        fit_poro_perm_transform(core)


# -- Core porosity correction under the "current" and "geolink" profile --------------


def _core_step(result):
    return next(step for step in result.steps_executed if step.step == "core_correction")


def test_current_fits_core_porosity_to_phit_and_takes_the_bound_part_off_again():
    # Plug helium porosity is total porosity: core = 1.1 PHIT - 0.02 exactly.
    well = _cored_well(_linear_core)
    phit = well.get("PHIT").data.copy()
    phie = well.get("PHIE").data.copy()

    result = DynamicWorkflow().run_selected_steps(well, [WorkflowStep.CORE_CORRECTION])
    step = _core_step(result)

    phit_cc = 1.1 * phit - 0.02
    expected = np.clip(phit_cc - (phit - phie), 0.0, phit_cc)
    np.testing.assert_allclose(result.well.get("PHIE_CC").data, expected, atol=1e-9)
    # PHIT 0.24, PHIE 0.21: PHIT_CC = 1.1 * 0.24 - 0.02 = 0.244, PHIE_CC = 0.244 - 0.03.
    assert result.well.get("PHIE_CC").data[15] == pytest.approx(0.214)
    # Shale, PHIT 0.06 and PHIE 0: PHIT_CC = 0.046 is all clay-bound, so PHIE_CC = 0.
    assert result.well.get("PHIE_CC").data[0] == pytest.approx(0.0)
    # The poro-perm transform is fitted on PHIT_CC: log10(k) = 12 PHIT_CC - 2.
    np.testing.assert_allclose(
        result.well.get("PERM_CC").data, np.minimum(10 ** (12.0 * phit_cc - 2.0), 10000.0), rtol=1e-6
    )
    fit = step.parameters["core_porosity_fits"][0]
    assert fit["input_curve"] == "PHIT"
    assert (fit["A"], fit["B"], fit["r_squared"], fit["n_points"]) == pytest.approx((1.1, -0.02, 1.0, 10))
    assert step.input_curves == ["PHIT", "PHIE"]
    assert "Core porosity correction (all depths): PHIT_CC = 1.100 * PHIT - 0.020 (n=10, r^2=1.00)" in step.decisions
    assert any(note.startswith("PHIE_CC = PHIT_CC - (PHIT - PHIE)") for note in step.decisions)


def _noisy_core(phit):
    # A weak trend under alternating scatter: positive slope, r^2 well below 0.5.
    return 0.10 + 0.5 * phit + 0.08 * (-1.0) ** np.arange(phit.size)


@pytest.mark.parametrize(
    "core_phi, n_core, reason",
    [
        (lambda phit: 0.40 - phit, 10, "non_positive_porosity_slope"),
        (_noisy_core, 10, "low_porosity_r_squared"),
        (_linear_core, 5, "too_few_porosity_points"),
    ],
)
def test_current_does_not_apply_an_unphysical_core_porosity_fit(core_phi, n_core, reason):
    well = _cored_well(core_phi, n_core=n_core)
    if reason == "low_porosity_r_squared":
        cored = np.isfinite(well.get("PHIC").data)
        x, y = well.get("PHIT").data[cored], well.get("PHIC").data[cored]
        assert np.polyfit(x, y, 1)[0] > 0 and np.corrcoef(x, y)[0, 1] ** 2 < 0.5

    result = DynamicWorkflow().run_selected_steps(well, [WorkflowStep.CORE_CORRECTION])
    step = _core_step(result)

    assert result.well.get("PHIE_CC") is None
    assert result.well.get("PERM_CC") is None
    rejected = step.parameters["core_rejected_models"]
    assert [item["reason"] for item in rejected] == [reason]
    assert rejected[0]["input_curve"] == "PHIT"
    assert any(
        note.startswith("Core porosity correction (all depths) not used: ") for note in step.decisions
    )


# The eval package's main run and its re-run (its re-run wells and context).
@pytest.mark.parametrize("calculations", ["main_run", "rerun"])
def test_benchmark_keeps_the_ungated_phie_core_fit(calculations):
    from petrophys.workflow.dynamic_workflow import _rerun_calculations

    # Core porosity falls as log porosity rises; the "geolink" profile still applies the fit.
    well = _cored_well(lambda phit: 0.40 - phit)
    phie = well.get("PHIE").data.copy()
    core = well.get("PHIC").data.copy()
    valid = np.isfinite(core) & (core > 0.01) & (phie > 0.01)
    a, b = np.polyfit(phie[valid], core[valid], 1)

    with _rerun_calculations(calculations == "rerun"):
        result = DynamicWorkflow(DynamicWorkflowConfig(calculation_profile="geolink")).run_selected_steps(
            well, [WorkflowStep.CORE_CORRECTION]
        )
    step = _core_step(result)

    assert a < 0
    np.testing.assert_allclose(result.well.get("PHIE_CC").data, np.clip(a * phie + b, 0.0, 0.5), atol=1e-9)
    assert step.input_curves == ["PHIE"]
    assert "core_porosity_fits" not in step.parameters
    assert step.parameters["core_rejected_models"] == []
    assert not any("Core porosity correction" in note for note in step.decisions)


@pytest.mark.parametrize("profile", ["current", "geolink"])
def test_context_notes_name_the_porosity_the_core_correction_was_fitted_to(profile):
    from petrophys.reporting.reference_export import _compact_curve_note
    from petrophys.workflow.dynamic_workflow import _rerun_calculations

    # The context export runs the "geolink" profile with the re-run's calculations.
    with _rerun_calculations():
        result = DynamicWorkflow(DynamicWorkflowConfig(calculation_profile=profile)).run_selected_steps(
            _cored_well(_linear_core), [WorkflowStep.CORE_CORRECTION]
        )
    step = _core_step(result)
    phie_note = _compact_curve_note("PHIE_CC", "computed", "", step, result.well)
    perm_note = _compact_curve_note("PERM_CC", "computed", "", step, result.well)
    if profile == "current":
        assert phie_note.startswith(
            "PHIE_CC: same-well core porosity correction fitted to PHIT; "
            "PHIT_CC=A*PHIT+B, PHIE_CC=clip(PHIT_CC-(PHIT-PHIE),0,PHIT_CC)."
        )
        assert "PERM_CC=10^(A2*PHIT_CC+B2)" in perm_note
    else:
        assert phie_note.startswith("PHIE_CC: same-well core porosity correction; PHIE_CC=A*PHIE+B")
        assert "PERM_CC=10^(A2*PHIE_CC+B2)" in perm_note


def test_permeability_module_core_porosity_options():
    well = _cored_well(_linear_core)
    phit = well.get("PHIT").data.copy()

    gated = PermeabilityModule(core_porosity_curve="PHIT", min_porosity_r_squared=0.5)
    result = gated.calculate_core_correction(well)
    # The module returns the core-equivalent PHIT_CC; the workflow takes PHIT - PHIE off.
    np.testing.assert_allclose(result.metadata["phi_corecal"], 1.1 * phit - 0.02, atol=1e-9)
    assert result.metadata["core_models"][0]["input_curve"] == "PHIT"

    # Without the options the correction is fitted to PHIE and every fit is used.
    default = PermeabilityModule().calculate_core_correction(_cored_well(lambda phit: 0.40 - phit))
    assert default.metadata["core_models"][0]["input_curve"] == "PHIE"
    assert default.metadata["core_porosity_fits"][0]["A"] < 0
    gated_result = gated.calculate_core_correction(_cored_well(lambda phit: 0.40 - phit))
    assert gated_result.metadata["phi_corecal"] is None
    assert gated_result.metadata["core_rejected_models"][0]["reason"] == "non_positive_porosity_slope"
