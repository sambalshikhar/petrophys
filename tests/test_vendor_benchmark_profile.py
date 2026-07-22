import numpy as np

from petrophys import Well
from petrophys.interpretation.saturation.indonesia import indonesia
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow._vendor_benchmark import may_2026_indonesia


def _profile_well(with_vendor_lithology: bool) -> Well:
    n_samples = 40
    well = Well("PROFILE-TEST")
    well.depth = np.arange(8000.0, 8020.0, 0.5)
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(25.0, 125.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.32, n_samples), "v/v", "Neutron")
    well.add_curve("DTC", np.linspace(70.0, 115.0, n_samples), "us/ft", "Sonic")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n_samples), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(50.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    if with_vendor_lithology:
        well.add_curve(
            "LITHOLOGY_GEOLINK",
            np.array([1.0] * 28 + [8.0] * 12),
            "",
            "Vendor lithology code",
        )
    return well


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _run(well, profile="auto"):
    config = DynamicWorkflowConfig(
        interpretation_profile=profile,
        standardize_input=False,
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
    )
    return DynamicWorkflow(config).run(well)


def test_auto_profile_uses_versioned_benchmark_for_vendor_lithology():
    well = _profile_well(with_vendor_lithology=True)
    result = _run(well)

    assert result.success
    assert result.interpretation_profile == "vendor_benchmark"
    assert _step(result, "vclay").parameters["nd_matrix"] == "limestone"
    assert _step(result, "lithology").method == "may_2026_multi_mineral_solver"
    assert _step(result, "saturation").parameters["vcl_cutoff_for_routing"] == 0.3
    assert _step(result, "net_pay").parameters["cutoff_source"] == "may_2026_benchmark"
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("LITHO") is None
    assert well.get("MMQC") is None
    assert "porosity_refinement" in result.steps_skipped
    assert result.to_trace_dict()["interpretation_profile"] == "vendor_benchmark"


def test_generic_override_keeps_current_solver_with_vendor_lithology():
    well = _profile_well(with_vendor_lithology=True)
    result = _run(well, profile="generic_inference")

    assert result.success
    assert result.interpretation_profile == "generic_inference"
    assert _step(result, "vclay").parameters["nd_matrix"] == "sandstone"
    assert _step(result, "lithology").method == "multi_mineral_solver"
    assert well.get("MMQC") is not None
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("LITHO") is None


def test_auto_profile_keeps_iterative_generic_route_without_vendor_lithology():
    well = _profile_well(with_vendor_lithology=False)
    result = _run(well)

    assert result.success
    assert result.interpretation_profile == "generic_inference"
    assert _step(result, "lithology").method == "multi_mineral_solver"
    assert well.get("LITHO") is not None
    assert well.get("MMQC") is not None
    assert any(step.step == "porosity_refinement" for step in result.steps_executed)


def test_forced_benchmark_without_vendor_lithology_falls_back_to_generic():
    result = _run(
        _profile_well(with_vendor_lithology=False),
        profile="vendor_benchmark",
    )

    assert result.interpretation_profile == "generic_inference"
    assert any("without a validated vendor lithology" in item for item in result.warnings)


def test_vendor_benchmark_without_neutron_skips_only_mineral_inversion():
    well = _profile_well(with_vendor_lithology=True)
    well.remove_curve("NPHI")

    result = _run(well)

    assert result.success
    assert result.interpretation_profile == "vendor_benchmark"
    assert "lithology" in result.steps_skipped
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("VQUAR") is None
    assert well.get("PHIE") is not None


def test_historical_indonesia_equation_is_isolated_from_generic_equation():
    arguments = {
        "rt": np.array([8.0, 12.0]),
        "rw": 0.05,
        "phi": np.array([0.18, 0.24]),
        "vcl": np.array([0.35, 0.50]),
        "rcl": 2.0,
    }

    historical = may_2026_indonesia(**arguments)
    generic = indonesia(**arguments)

    assert np.all(np.isfinite(historical))
    assert not np.allclose(historical, generic)
