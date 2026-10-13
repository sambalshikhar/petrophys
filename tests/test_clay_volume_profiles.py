"""Clay volume in the default profile, and the "geolink" profile kept as it was.

The default ("current") profile measures VCL_ND from the clean line of the
resolved matrix and a clay point picked in good-hole shales of the same well,
and keeps an SP with whole-log baselines out of the minimum-of-indicators VCL.
The "geolink" profile keeps the limestone clean line, the fixed clay point
(2.62 g/cm3, 0.38) and SP in the minimum, with the calculations of the eval
package's main run and of its re-run alike. Expected values are worked by hand
from the three-point neutron-density formula and the SP and GR indices.
"""
import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.clay_volume import VclModule, vcl_from_nd, vcl_from_sp
from petrophys.interpretation.clay_volume.nd_method import nd_clay_separation, nd_clean_line, nd_shale_point
from petrophys.interpretation.clay_volume.sp_method import sp_polarity
from petrophys.reporting.reference_export import _vcl_curve_note
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult, StepResult, _rerun_calculations
from petrophys.workflow.zoned_batch import _finalize_export_curves, run_dynamic_workflow_with_zones

#: The calculation sets of the "geolink" profile: the eval package's main run,
#: and its re-run, which its re-run wells and its context export get.
BENCHMARK_CALCULATIONS = ("main_run", "rerun")

# Sandstone line (2.65, -0.02)-(1.99, 0.38) and a shale point (2.35, 0.40):
# a shaly sand at RHOB 2.40, NPHI 0.25 lies
#   ((1.99-2.65)*(0.25+0.02) - (2.40-2.65)*(0.38+0.02))
#   / ((1.99-2.65)*(0.40+0.02) - (2.35-2.65)*(0.38+0.02)) = 0.0782/0.1572
# of the way from the clean line to the shale point.
SHALY_SAND_CURRENT = 0.0782 / 0.1572
# The same sample against the limestone line (2.71, 0)-(2.03, 0.40) and the
# fixed clay point (2.62, 0.38): 0.046/0.2224.
SHALY_SAND_BENCHMARK = 0.046 / 0.2224


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


# ---------------------------------------------------------------------------
# Low-level functions
# ---------------------------------------------------------------------------
def test_shale_point_known_value():
    nphi, rhob = np.array([0.25]), np.array([2.40])
    clean = nd_clean_line("sandstone")
    value = vcl_from_nd(nphi, rhob, *clean, den_clay=2.35, neu_clay=0.40)
    assert float(value[0]) == pytest.approx(SHALY_SAND_CURRENT, rel=1e-9)
    # The shale point itself reads 1, and a clean 25 p.u. sandstone
    # (RHOB = 2.65 - 0.25*1.65, NPHI on the line) reads 0.
    assert vcl_from_nd(0.40, 2.35, *clean, den_clay=2.35, neu_clay=0.40) == pytest.approx(1.0)
    assert vcl_from_nd(0.23, 2.2375, *clean, den_clay=2.35, neu_clay=0.40) == pytest.approx(0.0, abs=1e-12)


def test_nd_shale_point_is_the_median_of_the_flagged_samples():
    rhob = np.r_[np.full(20, 2.33), np.full(20, 2.37), np.full(6, 2.00), 2.20, np.nan]
    nphi = np.r_[np.full(20, 0.38), np.full(20, 0.42), np.full(6, 0.55), 0.10, 0.40]
    shale = np.r_[np.ones(40, bool), np.zeros(6, bool), False, True]
    den, neu = nd_shale_point(nphi, rhob, shale, matrix="sandstone")
    assert den == pytest.approx(2.35)
    assert neu == pytest.approx(0.40)


def test_nd_shale_point_needs_enough_samples():
    with pytest.raises(ValueError, match="19 shale samples"):
        nd_shale_point(np.full(19, 0.40), np.full(19, 2.35), np.ones(19, bool), matrix="sandstone")


def test_nd_shale_point_must_lie_clear_of_the_clean_line():
    # (2.32, 0.18) lies on the sandstone line: -0.02 + (2.65 - 2.32)*0.40/0.66.
    with pytest.raises(ValueError, match="clean line"):
        nd_shale_point(np.full(30, 0.18), np.full(30, 2.32), np.ones(30, bool), matrix="sandstone")


def test_nd_clean_line_accepts_matrix_property_aliases():
    for name in ("ss", "silica", "sio2"):
        assert nd_clean_line(name) == nd_clean_line("sandstone")
    assert nd_clean_line("dolo") == nd_clean_line("dolomite")
    for name in ("lime", "ls", "calcium_carbonate"):
        assert nd_clean_line(name) == nd_clean_line("limestone")


def test_sp_baselines_mean_clean_line_and_shale_baseline():
    # sp_max is the clean-sand line, sp_min the shale baseline (normal SP).
    assert vcl_from_sp(-60.0, sp_max=-60.0, sp_min=-10.0) == pytest.approx(0.0)
    assert vcl_from_sp(-10.0, sp_max=-60.0, sp_min=-10.0) == pytest.approx(1.0)
    assert vcl_from_sp(-35.0, sp_max=-60.0, sp_min=-10.0) == pytest.approx(0.5)


def _bedded_logs(deflection: float):
    """1000 m of 5 m sand/shale beds; GR trends up and the SP drifts down."""
    depth = np.arange(0.0, 1000.0, 0.5)
    sand = (depth // 5.0) % 2 == 0
    gr = np.where(sand, 30.0, 120.0) + 0.1 * depth
    sp = np.where(sand, deflection, 0.0) - 0.2 * depth
    return sp, gr, depth


def test_sp_polarity_reads_the_deflection_through_baseline_drift():
    sp, gr, depth = _bedded_logs(-30.0)
    # The drift alone would make SP fall as GR rises over the whole log.
    assert np.corrcoef(sp, gr)[0, 1] < 0
    assert sp_polarity(sp, gr, depth) == 1
    sp, gr, depth = _bedded_logs(30.0)
    assert sp_polarity(sp, gr, depth) == -1


def test_sp_polarity_is_unclear_without_a_bed_signal():
    rng = np.random.default_rng(7)
    depth = np.arange(0.0, 1000.0, 0.5)
    gr = np.where((depth // 5.0) % 2 == 0, 30.0, 120.0)
    assert sp_polarity(rng.normal(0.0, 5.0, depth.size), gr, depth) == 0
    assert sp_polarity(np.full(depth.size, -20.0), gr, depth) == 0
    assert sp_polarity(gr[:40], gr[:40], depth[:40]) == 0


def _indicator_well() -> Well:
    well = Well("INDICATORS")
    well.depth = np.arange(1000.0, 1004.0)
    well.depth_unit = "m"
    well.add_curve("GR", np.array([30.0, 80.0, 120.0, np.nan]), "API", "Gamma Ray")
    well.add_curve("SP", np.array([-60.0, -35.0, -40.0, -35.0]), "mV", "SP")
    well.add_curve("NPHI", np.array([0.23, 0.25, 0.40, np.nan]), "v/v", "Neutron")
    well.add_curve("RHOB", np.array([2.2375, 2.40, 2.35, np.nan]), "g/cc", "Density")
    return well


def test_sp_fallback_only_keeps_sp_out_of_the_minimum():
    settings = dict(gr_min=20.0, gr_max=120.0, sp_max=-60.0, sp_min=-10.0, matrix="sandstone",
                    nd_clay=(2.35, 0.40))
    both = VclModule(**settings).calculate(_indicator_well())
    fallback = VclModule(**settings, sp_fallback_only=True).calculate(_indicator_well())

    # VCL_GR = (GR-20)/100, VCL_SP = (-60-SP)/(-60+10).
    np.testing.assert_allclose(fallback.vcl_sp, [0.0, 0.5, 0.4, 0.5])
    # The shale sample: SP (0.4) pulls the minimum below GR and N-D (1.0).
    assert both.vcl[2] == pytest.approx(0.4)
    assert fallback.vcl[2] == pytest.approx(1.0)
    assert fallback.vcl[1] == pytest.approx(SHALY_SAND_CURRENT)
    # Only SP reads at the last depth, so it fills that one.
    assert fallback.vcl[3] == pytest.approx(0.5) and fallback.method_used[3] == "SP"
    assert list(fallback.method_used[:3]) == ["ND", "ND", "GR"]


def test_vcl_module_defaults_are_unchanged():
    module = VclModule()
    assert module.sp_fallback_only is False and module.nd_clay is None and module.nd_triangle_only is False
    well = _indicator_well()
    result = VclModule(gr_min=20.0, gr_max=120.0, sp_max=-60.0, sp_min=-10.0).calculate(well)
    # Limestone line and the fixed clay point.
    assert result.vcl_nd[1] == pytest.approx(SHALY_SAND_BENCHMARK)
    assert result.vcl[2] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Workflow: neutron-density clean line and clay point
# ---------------------------------------------------------------------------
def _nd_well(n_good_shale: int = 40, lithology: bool = True) -> Well:
    """Sand and shale with known logs; six washed-out shale samples read light.

    Good-hole shale: half at (2.33, 0.38), half at (2.37, 0.42), median
    (2.35, 0.40). With the washouts (2.00, 0.55) the median would move to
    (2.33, 0.42).
    """
    half = n_good_shale // 2
    rhob = np.r_[np.full(half, 2.33), np.full(half, 2.37), np.full(6, 2.00), np.full(30, 2.2375), np.full(4, 2.40)]
    nphi = np.r_[np.full(half, 0.38), np.full(half, 0.42), np.full(6, 0.55), np.full(30, 0.23), np.full(4, 0.25)]
    gr = np.r_[np.full(n_good_shale + 6, 120.0), np.full(30, 30.0), np.full(4, 80.0)]
    cali = np.r_[np.full(n_good_shale, 8.6), np.full(6, 12.0), np.full(34, 8.6)]
    codes = np.r_[np.full(n_good_shale + 6, 8.0), np.full(34, 1.0)]
    well = Well("ND-CLAY-POINT")
    well.depth = 2000.0 + 0.5 * np.arange(rhob.size)
    well.depth_unit = "m"
    well.add_curve("GR", gr, "API", "Gamma Ray")
    well.add_curve("RHOB", rhob, "g/cc", "Bulk Density")
    well.add_curve("NPHI", nphi, "v/v", "Neutron")
    well.add_curve("CALI", cali, "in", "Caliper")
    well.add_curve("BS", np.full(rhob.size, 8.5), "in", "Bit Size")
    if lithology:
        well.add_curve("LITHOLOGY_GEOLINK", codes, "", "Lithology")
    return well


def _run(well: Well, **settings):
    config = DynamicWorkflowConfig(rw=0.05, gr_min=20.0, gr_max=120.0, **settings)
    return DynamicWorkflow(config).run(well)


def _benchmark_run(well: Well, calculations: str, **settings):
    """Run the "geolink" profile with the main run's or the re-run's calculations."""
    with _rerun_calculations(calculations == "rerun"):
        return _run(well, calculation_profile="geolink", **settings)


def test_current_uses_the_matrix_line_and_the_shale_point_of_the_well():
    well = _nd_well()
    result = _run(well)
    params = _step(result, "vclay").parameters

    assert params["nd_matrix"] == "sandstone"
    assert (params["nd_clean1_density"], params["nd_clean1_neutron"]) == (2.65, -0.02)
    assert params["nd_clay_source"] == "shale_samples"
    assert params["nd_clay_basis"] == "lithology_shale"
    # The six washouts are bad hole and stay out of the pick.
    assert params["nd_clay_samples"] == 40
    assert params["nd_clay_density"] == pytest.approx(2.35)
    assert params["nd_clay_neutron"] == pytest.approx(0.40)

    vcl = np.asarray(well["VCL"].data)
    # Shaly sand: min(VCL_GR = 0.6, VCL_ND = 0.497).
    np.testing.assert_allclose(vcl[-4:], SHALY_SAND_CURRENT, rtol=1e-6)
    # Clean sandstone sits on the sandstone line.
    np.testing.assert_allclose(vcl[46:76], 0.0, atol=1e-9)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_keeps_the_limestone_line_and_fixed_clay_point(calculations):
    well = _nd_well()
    step = _step(_benchmark_run(well, calculations), "vclay")
    params = step.parameters

    assert params["nd_matrix"] == "limestone"
    assert (params["nd_clean1_density"], params["nd_clean1_neutron"]) == (2.71, 0.0)
    assert (params["nd_clay_density"], params["nd_clay_neutron"]) == (2.62, 0.38)
    assert not {"sp_in_vcl", "sp_polarity", "nd_clay_source"} & set(params)
    assert step.decisions[-2].startswith("Final VCL = min(available methods), range: ")
    assert step.decisions[-1] == "Canonical limestone N-D endpoints used"
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[-4:], SHALY_SAND_BENCHMARK, rtol=1e-6)


@pytest.mark.parametrize(
    "matrix, line",
    [("sandstone", (2.65, -0.02)), ("limestone", (2.71, 0.0)), ("dolomite", (2.87, 0.02))],
)
def test_current_clean_line_follows_the_matrix(matrix, line):
    domain = "clastic" if matrix == "sandstone" else "carbonate"
    params = _step(_run(_nd_well(), matrix_type=matrix, lithology_domain=domain), "vclay").parameters
    assert params["nd_matrix"] == matrix
    assert (params["nd_clean1_density"], params["nd_clean1_neutron"]) == line


def test_without_a_lithology_log_shales_are_picked_by_gamma_ray():
    params = _step(_run(_nd_well(lithology=False)), "vclay").parameters
    assert params["nd_clay_basis"] == "vcl_gr_gt_0.9"
    assert params["nd_clay_source"] == "shale_samples"
    assert params["nd_clay_samples"] == 40
    assert params["nd_clay_density"] == pytest.approx(2.35)


def test_too_few_shale_samples_keep_the_default_clay_point():
    result = _run(_nd_well(n_good_shale=12))
    step = _step(result, "vclay")
    assert step.parameters["nd_clay_source"] == "default"
    assert (step.parameters["nd_clay_density"], step.parameters["nd_clay_neutron"]) == (2.62, 0.38)
    reason = step.parameters["nd_clay_fallback_reason"]
    assert "12 good-hole shale-lithology samples" in reason and "12 good-hole VCL_GR > 0.9 samples" in reason
    assert any(d.startswith("N-D clay point: default") for d in step.decisions)


# ---------------------------------------------------------------------------
# Workflow: SP
# ---------------------------------------------------------------------------
def _sp_well() -> Well:
    """100 samples of 5 m sand/shale beds with a normal SP, and one shaly sand.

    Sand: GR 30, SP -60; shale: GR 120, SP -10; the shaly sand (sample 4) has
    GR 80 (VCL_GR 0.6) and SP -35 (VCL_SP 0.5 between -60 and -10).
    """
    n = 100
    sand = (np.arange(n) // 10) % 2 == 0
    gr = np.where(sand, 30.0, 120.0)
    sp = np.where(sand, -60.0, -10.0)
    gr[4], sp[4] = 80.0, -35.0
    well = Well("SP-WELL")
    well.depth = 1500.0 + 0.5 * np.arange(n)
    well.depth_unit = "m"
    well.add_curve("GR", gr, "API", "Gamma Ray")
    well.add_curve("SP", sp, "mV", "SP")
    return well


def test_current_reports_whole_log_sp_but_keeps_it_out_of_vcl():
    result = _run(_sp_well())
    step = _step(result, "vclay")
    well = result.well

    assert step.parameters["use_sp"] is True
    assert step.parameters["sp_in_vcl"] is False
    assert well["VCL_SP"].data[4] == pytest.approx(0.5)
    assert well["VCL"].data[4] == pytest.approx(0.6)
    assert any("VCL_SP is reported but left out of the minimum" in d for d in step.decisions)
    assert "final=min(GR); " in _vcl_curve_note(step) and "(reported only)" in _vcl_curve_note(step)

    # An export recombining the indicator curves leaves the reported SP out.
    _finalize_export_curves(well, {"GR", "SP"})
    assert well["VCL"].data[4] == pytest.approx(0.6)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_keeps_sp_in_the_minimum(calculations):
    result = _benchmark_run(_sp_well(), calculations)
    step = _step(result, "vclay")
    well = result.well

    assert well["VCL"].data[4] == pytest.approx(0.5)
    assert "SP_clean = -60.0 (auto, P5 of SP)" in step.decisions
    assert "final=min(GR, SP); " in _vcl_curve_note(step)
    _finalize_export_curves(well, {"GR", "SP"})
    assert well["VCL"].data[4] == pytest.approx(0.5)


def test_user_sp_baselines_join_the_minimum():
    result = _run(_sp_well(), sp_min=-10.0, sp_max=-60.0)
    step = _step(result, "vclay")
    assert step.parameters["sp_in_vcl"] is True
    assert step.parameters["sp_polarity"] == "normal"
    assert result.well["VCL"].data[4] == pytest.approx(0.5)
    assert "SP_clean = -60.0 (user)" in step.decisions
    assert not any("swapped" in warning for warning in result.warnings)


def test_one_user_sp_baseline_keeps_sp_reported_only():
    # Only the shale baseline is given; the clean line is still the whole-log P5.
    result = _run(_sp_well(), sp_min=-10.0)
    step = _step(result, "vclay")
    assert step.parameters["sp_in_vcl"] is False
    assert step.parameters["sp_polarity"] == "normal"
    assert "SP_clean = -60.0 (auto, P5 of SP)" in step.decisions
    assert "SP_shale = -10.0 (user)" in step.decisions
    assert result.well["VCL"].data[4] == pytest.approx(0.6)
    assert any("Set both sp_min and sp_max" in d for d in step.decisions)


def test_swapped_sp_baselines_are_flagged():
    # Shale baseline and clean line exchanged: VCL_SP reads 1 in clean sand.
    result = _run(_sp_well(), sp_min=-60.0, sp_max=-10.0)
    assert result.well["VCL_SP"].data[0] == pytest.approx(1.0)
    assert any("check whether the two values are swapped" in w for w in result.warnings)
    assert any("contradicts" in d for d in _step(result, "vclay").decisions)


def test_equal_sp_baselines_are_rejected_in_the_default_profile():
    with pytest.raises(ValueError, match="sp_min .* must differ"):
        DynamicWorkflowConfig(sp_min=-20.0, sp_max=-20.0)
    DynamicWorkflowConfig(sp_min=-20.0, sp_max=-20.0, calculation_profile="geolink")


# ---------------------------------------------------------------------------
# Neutron-density triangle, clay point fall-through, unusable clay points
# ---------------------------------------------------------------------------
def test_nd_clay_separation_known_values():
    # Sandstone line at 2.35 g/cm3: -0.02 + (2.35 - 2.65) * 0.40 / (1.99 - 2.65)
    # = 0.1618, so (2.35, 0.40) lies 0.2382 above it; (2.32, 0.18) lies on it.
    assert nd_clay_separation(2.35, 0.40, "sandstone") == pytest.approx(0.40 - (-0.02 + 0.30 * 0.40 / 0.66))
    assert nd_clay_separation(2.32, 0.18, "sandstone") == pytest.approx(0.0, abs=1e-12)
    # Limestone line at 2.62 g/cm3: (2.62 - 2.71) * 0.40 / (2.03 - 2.71) = 0.0529.
    assert nd_clay_separation(2.62, 0.38) == pytest.approx(0.38 - 0.09 * 0.40 / 0.68)


def _mudstone_well() -> Well:
    """Clean sand, an undercompacted mudstone, the shale point, a shaly sand."""
    well = Well("TRIANGLE")
    well.depth = np.arange(1000.0, 1004.0)
    well.depth_unit = "m"
    well.add_curve("GR", np.array([30.0, 100.0, 120.0, 80.0]), "API", "Gamma Ray")
    well.add_curve("NPHI", np.array([0.23, 0.45, 0.40, 0.25]), "v/v", "Neutron")
    well.add_curve("RHOB", np.array([2.2375, 1.95, 2.35, 2.40]), "g/cc", "Density")
    return well


def test_vcl_nd_is_undefined_outside_the_triangle():
    settings = dict(gr_min=20.0, gr_max=120.0, use_sp=False, matrix="sandstone", nd_clay=(2.35, 0.40))
    plain = VclModule(**settings).calculate(_mudstone_well())
    triangle = VclModule(**settings, nd_triangle_only=True).calculate(_mudstone_well())

    # The mudstone (1.95 g/cm3, NPHI 0.45) lies beyond the clean line's end
    # point (0.38) and the clay point (0.40): the formula gives
    # ((1.99-2.65)*(0.45+0.02) - (1.95-2.65)*0.40) / -0.1572 = 0.0302/0.1572.
    assert plain.vcl_nd[1] == pytest.approx(0.0302 / 0.1572)
    assert plain.vcl[1] == pytest.approx(0.0302 / 0.1572)
    assert np.isnan(triangle.vcl_nd[1])
    assert triangle.vcl[1] == pytest.approx(0.8)  # VCL_GR = (100 - 20) / 100
    # On the triangle's edge (the clay point) and inside it nothing changes.
    np.testing.assert_allclose(triangle.vcl_nd[[0, 2, 3]], [0.0, 1.0, SHALY_SAND_CURRENT], atol=1e-9)


def test_current_leaves_vcl_nd_undefined_beyond_the_triangle():
    well = _nd_well()
    step = _step(_run(well), "vclay")
    # NPHI above max(0.38, 0.40): the 20 shale samples at 0.42 and the six
    # washouts at 0.55. The shales take VCL_GR (1.0) there.
    vcl_nd = np.asarray(well["VCL_ND"].data)
    assert np.isnan(vcl_nd[20:46]).all() and np.isfinite(vcl_nd[:20]).all()
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[20:40], 1.0)
    assert step.parameters["nd_outside_triangle_samples"] == 26
    assert any(d.startswith("VCL_ND undefined at 26 samples with NPHI above 0.400") for d in step.decisions)

    for calculations in BENCHMARK_CALCULATIONS:
        well = _nd_well()
        _benchmark_run(well, calculations)
        assert np.isfinite(np.asarray(well["VCL_ND"].data)[20:46]).all()


def test_too_few_lithology_shales_fall_through_to_gamma_ray_shales():
    # Only 12 shale-coded samples (the rest of the shale is coded silt): the
    # pick moves on to the 40 good-hole VCL_GR > 0.9 samples, median (2.35, 0.40).
    well = _nd_well()
    codes = np.asarray(well["LITHOLOGY_GEOLINK"].data).copy()
    codes[12:46] = 5.0
    well.add_curve("LITHOLOGY_GEOLINK", codes, "", "Lithology")
    step = _step(_run(well), "vclay")
    params = step.parameters

    assert (params["nd_clay_source"], params["nd_clay_basis"], params["nd_clay_samples"]) == (
        "shale_samples", "vcl_gr_gt_0.9", 40,
    )
    assert (params["nd_clay_density"], params["nd_clay_neutron"]) == pytest.approx((2.35, 0.40))
    assert params["nd_clay_fallback_reason"] == (
        "12 good-hole shale-lithology samples have neutron and density; 20 are needed"
    )
    assert params["nd_clay_depth_range"] == [2000.0, 2019.5]
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[-4:], SHALY_SAND_CURRENT, rtol=1e-6)


def _clean_line_shale_well() -> Well:
    """Shale whose logs plot on the sandstone clean line, (2.32 g/cm3, 0.18)."""
    rhob = np.r_[np.full(30, 2.32), np.full(30, 2.2375), np.full(4, 2.40)]
    nphi = np.r_[np.full(30, 0.18), np.full(30, 0.23), np.full(4, 0.25)]
    gr = np.r_[np.full(30, 120.0), np.full(30, 30.0), np.full(4, 80.0)]
    well = Well("ON-THE-LINE")
    well.depth = 2000.0 + 0.5 * np.arange(rhob.size)
    well.depth_unit = "m"
    well.add_curve("GR", gr, "API", "Gamma Ray")
    well.add_curve("RHOB", rhob, "g/cc", "Bulk Density")
    well.add_curve("NPHI", nphi, "v/v", "Neutron")
    well.add_curve("CALI", np.full(rhob.size, 8.6), "in", "Caliper")
    well.add_curve("BS", np.full(rhob.size, 8.5), "in", "Bit Size")
    well.add_curve("LITHOLOGY_GEOLINK", np.r_[np.full(30, 8.0), np.full(34, 1.0)], "", "Lithology")
    return well


def test_a_clay_point_on_the_clean_line_leaves_vcl_nd_out_of_vcl():
    well = _clean_line_shale_well()
    result = _run(well)
    step = _step(result, "vclay")
    params = step.parameters

    assert params["nd_in_vcl"] is False
    assert params["nd_clay_picked"] == pytest.approx([2.32, 0.18])
    assert (params["nd_clay_density"], params["nd_clay_neutron"]) == (2.62, 0.38)
    assert params["nd_clay_fallback_reason"].startswith("the clay point (2.320 g/cm3, 0.180 v/v) lies ")
    assert params["nd_clay_fallback_reason"].endswith("above the sandstone clean line, less than the 0.10 needed")
    # VCL_ND, on the default point, reads the shale as clean (0); VCL is VCL_GR.
    vcl_nd, vcl = np.asarray(well["VCL_ND"].data), np.asarray(well["VCL"].data)
    np.testing.assert_allclose(vcl_nd[:30], 0.0, atol=1e-9)
    np.testing.assert_allclose(vcl[[0, 30, 60]], [1.0, 0.1, 0.6])
    assert well["VCL_ND"].metadata.custom["in_vcl"] is False
    assert any(w.startswith("VCL_ND is reported but left out of VCL") for w in result.warnings)
    assert "Final VCL = VCL_GR at each depth, range: 0.100 - 1.000" in step.decisions
    note = _vcl_curve_note(step)
    assert "final=min(GR); " in note and "clay=(2.62,0.38) (reported only)" in note
    _finalize_export_curves(well, {"GR", "NPHI", "RHOB"})
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[:30], 1.0)

    for calculations in BENCHMARK_CALCULATIONS:
        benchmark = _benchmark_run(_clean_line_shale_well(), calculations)
        assert "nd_in_vcl" not in _step(benchmark, "vclay").parameters
        assert not any("left out of VCL" in w for w in benchmark.warnings)


def test_a_set_clay_point_is_used_under_current_only():
    # Point (2.40, 0.45) on the sandstone line: the shaly sand (2.40, 0.25) reads
    # ((1.99-2.65)*0.27 - (2.40-2.65)*0.40) / ((1.99-2.65)*0.47 - (2.40-2.65)*0.40)
    # = 0.0782/0.2102, below its VCL_GR of 0.6.
    well = _nd_well()
    step = _step(_run(well, nd_clay=(2.40, 0.45)), "vclay")
    assert (step.parameters["nd_clay_source"], step.parameters["nd_clay_density"]) == ("config", 2.40)
    assert "N-D clay point: RHOB 2.400 g/cm3, NPHI 0.450 (nd_clay)" in step.decisions
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[-4:], 0.0782 / 0.2102, rtol=1e-6)

    # A set point on the clean line is not used either.
    result = _run(_nd_well(), nd_clay=(2.32, 0.18))
    assert _step(result, "vclay").parameters["nd_in_vcl"] is False
    assert any(w.startswith("VCL_ND is reported but left out of VCL") for w in result.warnings)

    for calculations in BENCHMARK_CALCULATIONS:
        params = _step(_benchmark_run(_nd_well(), calculations, nd_clay=(2.40, 0.45)), "vclay").parameters
        assert (params["nd_clay_density"], params["nd_clay_neutron"]) == (2.62, 0.38)
    with pytest.raises(ValueError, match="nd_clay must be"):
        DynamicWorkflowConfig(nd_clay=(0.40, 2.35))


def test_zones_take_the_whole_well_clay_point():
    # A header zone over the sand alone holds no shale. On its own it would
    # fall back to (2.62, 0.38) and read the shaly sand at 0.3103; with the
    # whole-well point (2.35, 0.40) it reads 0.0782/0.1572 as the base run does.
    def zoned(profile):
        well = _nd_well()
        well.set_parameter("TOP_Z1", 2023.0)
        well.set_parameter("BOT_Z1", 2039.5)
        well.set_parameter("RW_Z1", 0.05)
        config = DynamicWorkflowConfig(rw=0.05, gr_min=20.0, gr_max=120.0, calculation_profile=profile)
        return well, run_dynamic_workflow_with_zones(well, config)

    well, result = zoned("current")
    zone = _step(result.zone_results[0], "vclay").parameters
    assert zone["nd_clay_source"] == "config"
    assert (zone["nd_clay_density"], zone["nd_clay_neutron"]) == pytest.approx((2.35, 0.40))
    np.testing.assert_allclose(np.asarray(well["VCL"].data)[-4:], SHALY_SAND_CURRENT, rtol=1e-6)

    for calculations in BENCHMARK_CALCULATIONS:
        with _rerun_calculations(calculations == "rerun"):
            _, result = zoned("geolink")
        assert result.zone_results[0].config.nd_clay is None
        zone = _step(result.zone_results[0], "vclay").parameters
        assert (zone["nd_clay_density"], zone["nd_clay_neutron"]) == (2.62, 0.38)


def test_an_unresolved_auto_matrix_uses_the_sandstone_line_and_says_so_when_it_resolves_otherwise():
    step = _step(_run(_nd_well(lithology=False), lithology_domain="auto", matrix_type="auto"), "vclay")
    assert step.parameters["nd_matrix"] == "sandstone"

    # The end of the run warns when the resolved matrix has another clean line.
    workflow = DynamicWorkflow(DynamicWorkflowConfig(matrix_type="limestone", lithology_domain="carbonate"))
    workflow._result = DynamicWorkflowResult(well=Well("X"), config=workflow.config)
    workflow._result.steps_executed.append(
        StepResult(step="vclay", success=True, parameters={"use_nd": True, "nd_matrix": "sandstone"})
    )
    workflow._warn_provisional_clean_line()
    assert workflow._result.warnings == [
        "The matrix was resolved (limestone) only after the clay-volume step, which measured VCL_ND "
        "from the provisional sandstone clean line. Set matrix_type='limestone' to measure it from "
        "the limestone line."
    ]
