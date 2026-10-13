import numpy as np
import pytest

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


def _run(well, profile="current", **settings):
    settings.setdefault("calculation_profile", profile)
    config = DynamicWorkflowConfig(
        standardize_input=False,
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
        **settings,
    )
    return DynamicWorkflow(config).run(well)


def test_canonical_route_is_used_with_embedded_lithology():
    well = _workflow_well(with_embedded_lithology=True)
    result = _run(well)

    assert result.success
    assert result.lithology_domain != "auto"
    assert _step(result, "vclay").parameters["nd_matrix"] == "sandstone"
    assert _step(result, "lithology").method == "canonical_multi_mineral_solver"
    assert _step(result, "saturation").parameters["vcl_cutoff_for_routing"] == 0.3
    # Net rock takes the clastic standard; the "geolink" profile keeps 0.3.
    assert _step(result, "net_pay").parameters["vcl_cutoff"] == 0.4
    benchmark = _run(_workflow_well(with_embedded_lithology=True), "geolink")
    assert _step(benchmark, "net_pay").parameters["vcl_cutoff"] == 0.3
    assert well.get("LITHOLOGY_GEOLINK") is not None
    assert well.get("LITHO") is None
    assert well.get("MMQC") is None
    assert "interpretation_profile" not in result.to_trace_dict()


def test_benchmark_keeps_the_limestone_nd_clean_line():
    result = _run(_workflow_well(with_embedded_lithology=True), calculation_profile="geolink")

    assert result.success
    assert _step(result, "vclay").parameters["nd_matrix"] == "limestone"


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


def test_canonical_indonesia_is_the_benchmark_sand_term_variant():
    """The "geolink" profile's Indonesia variant, against hand-computed values.

    Rt=10, Rw=0.05, phi=0.20, Vcl=0.30, Rcl=2, a=1, m=n=2. The variant solves
    1/sqrt(Rt) = clay + sand * Sw^(n/2); the published equation of Poupon and
    Leveaux (1971) solves 1/sqrt(Rt) = (clay + sand) * Sw^(n/2).
    """
    clay = 0.30 ** (1 - 0.30 / 2) / np.sqrt(2.0)  # 0.25411
    sand = 0.20 / np.sqrt(0.05)  # 0.89443
    arguments = {
        "rt": np.array([10.0]),
        "rw": 0.05,
        "phi": np.array([0.20]),
        "vcl": np.array([0.30]),
        "rcl": 2.0,
    }

    variant = canonical_indonesia(**arguments)[0]
    published = indonesia(**arguments)[0]
    assert variant == pytest.approx((1 / np.sqrt(10.0) - clay) / sand, rel=1e-12)
    assert variant == pytest.approx(0.0694, abs=1e-4)
    assert published == pytest.approx(1 / np.sqrt(10.0) / (clay + sand), rel=1e-12)
    assert published == pytest.approx(0.2753, abs=1e-4)
