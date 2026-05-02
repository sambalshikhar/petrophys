import numpy as np

from petrophys.base import Well
from petrophys.petrophysics.permeability.perm_module import PermeabilityModule
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
    core_phi = np.array([0.18, 0.11, 0.24, 0.13, 0.20])
    core_perm = np.array([10.0, 300.0, 2.0, 80.0, 6.0])
    well = _core_well(core_phi, core_perm)
    well.add_curve("PHIE_CC", np.full(well.n_samples, 0.30), "v/v", "Stale core porosity")
    well.add_curve("PERM_CC", np.full(well.n_samples, 5000.0), "mD", "Stale core perm")

    result = DynamicWorkflow().run_selected_steps(well, [WorkflowStep.PERMEABILITY])

    assert result.well.get("PERM") is not None
    assert result.well.get("PHIE_CC") is not None
    assert result.well.get("PERM_CC") is not None
    assert not np.allclose(result.well.get("PHIE_CC").data, 0.30)
    assert not np.allclose(result.well.get("PERM_CC").data, 5000.0)


def test_dynamic_workflow_builds_phie_cc_before_saturation():
    core_phi = np.array([0.18, 0.20, 0.22, 0.24, 0.26])
    core_perm = 10 ** (12.0 * core_phi - 2.0)
    well = _core_well(core_phi, core_perm)

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
