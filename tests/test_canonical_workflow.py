import numpy as np

from petrophys import Well
from petrophys.interpretation.saturation.indonesia import indonesia
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow._canonical_calculations import canonical_indonesia


def _workflow_well(with_embedded_lithology: bool) -> Well:
    n_samples = 40
    well = Well("CANONICAL-TEST")
    well.depth = np.arange(8000.0, 8020.0, 0.5)
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(25.0, 125.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.32, n_samples), "v/v", "Neutron")
    well.add_curve("DTC", np.linspace(70.0, 115.0, n_samples), "us/ft", "Sonic")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n_samples), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(50.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    if with_embedded_lithology:
        well.add_curve(
            "LITHOLOGY_GEOLINK",
            np.array([1.0] * 28 + [8.0] * 12),
            "",
            "Embedded lithology code",
        )
    return well


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _run(well):
    config = DynamicWorkflowConfig(
        standardize_input=False,
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
    )
    return DynamicWorkflow(config).run(well)


def test_canonical_route_is_used_with_embedded_lithology():
    well = _workflow_well(with_embedded_lithology=True)
    result = _run(well)

    assert result.success
    assert result.lithology_domain != "auto"
    assert _step(result, "vclay").parameters["nd_matrix"] == "limestone"
    assert _step(result, "lithology").method == "canonical_multi_mineral_solver"
    assert _step(result, "saturation").parameters["vcl_cutoff_for_routing"] == 0.3
    assert _step(result, "net_pay").parameters["vcl_cutoff"] == 0.3
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("LITHO") is None
    assert well.get("MMQC") is None
    assert "interpretation_profile" not in result.to_trace_dict()


def test_canonical_route_writes_computed_lithology_when_input_is_absent():
    well = _workflow_well(with_embedded_lithology=False)
    result = _run(well)
    lithology_step = _step(result, "lithology")

    assert result.success
    assert lithology_step.parameters["lithology_curve_written"] is True
    assert "LITHO" in lithology_step.output_curves
    assert well.get("LITHO") is not None
    assert well.get("MMQC") is None


def test_missing_neutron_skips_only_optional_mineral_inversion():
    well = _workflow_well(with_embedded_lithology=True)
    well.remove_curve("NPHI")
    result = _run(well)

    assert result.success
    assert "lithology" in result.steps_skipped
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("VQUAR") is None
    assert well.get("PHIE") is not None


def test_canonical_indonesia_operation_order_is_stable():
    arguments = {
        "rt": np.array([8.0, 12.0]),
        "rw": 0.05,
        "phi": np.array([0.18, 0.24]),
        "vcl": np.array([0.35, 0.50]),
        "rcl": 2.0,
    }

    canonical = canonical_indonesia(**arguments)
    generic = indonesia(**arguments)
    assert np.all(np.isfinite(canonical))
    assert not np.allclose(canonical, generic)
