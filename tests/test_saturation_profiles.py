"""Shaly-sand water saturation under the current and the "geolink" profile.

The ``"current"`` profile uses the published equations: the Indonesia equation
of Poupon and Leveaux (1971); the wet-zone Dual Water form of Clavier, Coates
and Dumanoir (1984), run only with a supplied Rwb; and Waxman and Smits (1968)
with CEC given in meq/100 g and its total-porosity Sw reported on the
effective basis. Oil-based mud writes no SXO or MHI. The ``"geolink"``
profile keeps the eval package's calculations, those of its main run and of
its re-run alike.

Expected values are computed by hand from the textbook equations.
"""
import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.interpretation.saturation.dual_water import dual_water
from petrophys.interpretation.saturation.indonesia import indonesia
from petrophys.interpretation.saturation.sw_module import (
    SaturationModule,
    _effective_from_total_sw,
    calculate_sw,
)
from petrophys.interpretation.saturation.waxman_smits import waxman_smits
from petrophys.reporting import reference_export as reference
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, Workflow, steps
from petrophys.workflow.dynamic_workflow import _rerun_calculations
from petrophys.workflow.zoned_batch import _extract_global_parameter_overrides
from petrophys.workflow._canonical_calculations import (
    CanonicalSaturationModule,
    canonical_dual_water,
    canonical_indonesia,
)

#: The calculation sets of the "geolink" profile: the eval package's main run,
#: and its re-run, which its re-run wells and its context export get.
BENCHMARK_CALCULATIONS = ["main_run", "rerun"]
SHALY = slice(0, 16)  # VCL 0.35, PHIE 0.20, PHIT 0.25, RDEP 10 ohm.m

# Keep the inputs as given: only Rcl, Rw, saturation, BVW and net pay run.
_SATURATION_ONLY = [
    "temperature", "bad_hole", "vclay", "rmf", "porosity", "lithology",
    "invasion", "permeability", "rock_quality", "mhi",
]


def _well() -> Well:
    """Sixteen shaly-sand samples over four shale samples (Rcl = 2 ohm.m)."""
    well = Well("SATURATION-PROFILES")
    well.depth = np.arange(2000.0, 2010.0, 0.5)
    well.depth_unit = "m"
    well.add_curve("VCL", np.r_[np.full(16, 0.35), np.full(4, 0.95)], "v/v", "Clay volume")
    well.add_curve("PHIE", np.r_[np.full(16, 0.20), np.full(4, 0.02)], "v/v", "Effective porosity")
    well.add_curve("PHIT", np.r_[np.full(16, 0.25), np.full(4, 0.16)], "v/v", "Total porosity")
    well.add_curve("RDEP", np.r_[np.full(16, 10.0), np.full(4, 2.0)], "ohm.m", "Deep resistivity")
    return well


def _run(profile: str, well: Well = None, *, calculations: str = "main_run", **settings):
    settings.setdefault("skip_steps", _SATURATION_ONLY)
    config = DynamicWorkflowConfig(rw=0.05, a=1.0, m=2.0, n=2.0, calculation_profile=profile, **settings)
    with _rerun_calculations(calculations == "rerun"):
        return DynamicWorkflow(config).run(well if well is not None else _well())


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _shaly(result, curve):
    return np.asarray(result.well.get(curve).data, dtype=float)[SHALY]


def _indonesia_terms(rw, vcl=0.35, phie=0.20, rcl=2.0):
    """Clay and sand conductance terms of the Indonesia equation, a = 1, m = 2."""
    return vcl ** (1 - vcl / 2) / np.sqrt(rcl), phie / np.sqrt(rw)


def _dual_water_swe(phit, rt, rw, rwb, vcl, phi_cl):
    """Clavier-Coates-Dumanoir Swe for a = 1, m = n = 2: the positive root of
    Cw Swt^2 + Swb (Cwb - Cw) Swt - 1/(PHIT^2 Rt) = 0, then (Swt - Swb)/(1 - Swb)."""
    swb = vcl * phi_cl / phit
    cw, cwb = 1 / np.asarray(rw, dtype=float), 1 / rwb
    b = swb * (cwb - cw)
    swt = (-b + np.sqrt(b ** 2 + 4 * cw / (phit ** 2 * rt))) / (2 * cw)
    return np.clip((swt - swb) / (1 - swb), 0.0, 1.0)


def _waxman_smits_n2(phit, rt, rw, cec_meq_per_g, b, rho_ma=2.65, m_star=2.0):
    """Closed-form root of Ct = (Sw^2 / F*) (Cw + B Qv / Sw) for n = 2."""
    f_star = phit ** -m_star
    qv = cec_meq_per_g * rho_ma * (1 - phit) / phit
    cw, ct = 1 / rw, 1 / rt
    # Cw Sw^2 + B Qv Sw - F* Ct = 0
    return (-b * qv + np.sqrt((b * qv) ** 2 + 4 * cw * f_star * ct)) / (2 * cw)


# -- The public equations ------------------------------------------------------


def test_dual_water_is_the_clavier_coates_dumanoir_form():
    # Clavier, Coates and Dumanoir (1984): Swb = Vcl*phi_cl/PHIT;
    # 1/Rt = PHIT^m/a Swt^n [Cw + (Swb/Swt)(Cwb - Cw)], for n = 2 the positive
    # root of Cw Swt^2 + Swb (Cwb - Cw) Swt - a/(PHIT^m Rt) = 0;
    # Swe = (Swt - Swb) / (1 - Swb). PHIT 0.25, Rt 2, Rw 0.05, Rwb 0.02,
    # Vcl 0.30, phi_cl 0.15.
    swb = 0.30 * 0.15 / 0.25  # 0.18
    cw, cwb = 1 / 0.05, 1 / 0.02  # 20, 50
    b = swb * (cwb - cw)  # 5.4
    k = 1 / (0.25 ** 2 * 2.0)  # 8
    swt = (-b + np.sqrt(b ** 2 + 4 * cw * k)) / (2 * cw)  # 0.51171
    expected = (swt - swb) / (1 - swb)
    assert expected == pytest.approx(0.4045, abs=1e-4)
    assert 0.25 ** 2 * swt ** 2 * (cw + swb / swt * (cwb - cw)) == pytest.approx(0.5)
    assert dual_water(0.25, 2.0, 0.05, 0.02, 0.30, 0.15) == pytest.approx(expected, rel=1e-12)


def test_dual_water_benchmark_variant_is_the_wet_zone_form_of_crain():
    # Crain's Petrophysical Handbook: Swb = Vcl*phi_cl/PHIT;
    # Ro = a Rw Rwb / (PHIT^m (Rwb + Swb (Rw - Rwb))); Swt = (Ro/Rt)^(1/n);
    # Swe = (Swt - Swb) / (1 - Swb). PHIT 0.25, Rt 10, Rw 0.05, Rwb 0.02,
    # Vcl 0.30, phi_cl 0.15.
    from petrophys._internal.benchmark import historical_benchmark

    swb = 0.30 * 0.15 / 0.25  # 0.18
    ro = 0.05 * 0.02 / (0.25 ** 2 * (0.02 + swb * (0.05 - 0.02)))  # 0.62992
    swt = np.sqrt(ro / 10.0)  # 0.25098
    expected = (swt - swb) / (1 - swb)
    assert expected == pytest.approx(0.0866, abs=1e-4)
    with historical_benchmark():
        assert dual_water(0.25, 10.0, 0.05, 0.02, 0.30, 0.15) == pytest.approx(expected, rel=1e-12)
    # Both forms agree in wet rock (Rt = Ro).
    assert dual_water(0.25, ro, 0.05, 0.02, 0.30, 0.15) == pytest.approx(1.0)


def test_dual_water_caps_bound_water_saturation_unlike_the_benchmark_variant():
    # Vcl * phi_cl = 0.135 exceeds PHIT = 0.05: the pore water is all clay-bound.
    arguments = dict(
        phi_t=np.array([0.05]), rt=np.array([5.0]), rw=0.05, rwb=0.015,
        vcl=np.array([0.9]), phi_cl=0.15,
    )
    assert dual_water(**arguments)[0] == pytest.approx(1.0)
    assert canonical_dual_water(**arguments)[0] == pytest.approx(0.0)


def test_waxman_smits_matches_the_closed_form_for_n_2():
    # CEC 5 meq/100 g = 0.05 meq/g, B 3.8 (S/m)/(meq/cm3), PHIT 0.20, Rt 10, Rw 0.05.
    expected = _waxman_smits_n2(0.20, 10.0, 0.05, 0.05, 3.8)
    assert expected == pytest.approx(0.3068, abs=1e-4)
    assert waxman_smits(0.20, 10.0, 0.05, 0.05, 2.65, 3.8) == pytest.approx(expected, abs=1e-6)


def test_saturation_module_rwb_and_waxman_smits_options():
    with pytest.raises(ValueError, match="rwb"):
        SaturationModule(model="dual_water", rwb_fallback=None).calculate(_well(), rw=0.05, rcl=2.0)
    # The "geolink" profile's module keeps Rwb = 0.3 Rw and the total-porosity Sw.
    module = CanonicalSaturationModule(rwb_fallback=None, effective_waxman_smits=True)
    assert module.rwb_fallback == 0.3
    assert module.effective_waxman_smits is False


def test_public_saturation_entry_points_do_not_guess_rwb():
    well = _well()
    assert SaturationModule().rwb_fallback is None
    with pytest.raises(ValueError, match="rwb"):
        calculate_sw(well, rw=0.05, rcl=2.0, model="dual_water")
    # Auto routing gives shaly carbonate without Rwb the clastic routing and
    # counts the samples.
    with pytest.warns(UserWarning, match="20 shaly carbonate samples used indonesia"):
        sw = calculate_sw(well, rw=0.05, rcl=2.0, lithology="carbonate")
    clay, sand = _indonesia_terms(0.05)
    np.testing.assert_allclose(sw[SHALY], (1 / np.sqrt(10.0)) / (clay + sand), rtol=1e-9)
    # A supplied Rwb runs Dual Water. With Rwb = 0.02 the bound water alone
    # (Swb = 0.21) conducts more than Rt = 10 allows, so Swe = 0; Rwb = 0.04
    # leaves free water.
    for rwb, swe_at_rwb in ((0.02, 0.0), (0.04, 0.0605)):
        expected = _dual_water_swe(0.25, 10.0, 0.05, rwb, 0.35, 0.15)
        assert expected == pytest.approx(swe_at_rwb, abs=1e-4)
        sw = calculate_sw(well, rw=0.05, rcl=2.0, lithology="carbonate", rwb=rwb, phi_cl=0.15)
        np.testing.assert_allclose(sw[SHALY], expected, rtol=1e-9, atol=1e-12)


def test_standard_workflow_runs_dual_water_only_with_rwb():
    from petrophys.workflow.standard_workflow import StandardWorkflow, WorkflowConfig

    def saturation(**settings):
        config = WorkflowConfig(
            rw=0.05, lithology="carbonate", vcl_cutoff=0.3, calculate_vcl=False,
            calculate_porosity=False, calculate_perm=False, calculate_net_pay=False, **settings,
        )
        with pytest.warns(DeprecationWarning):
            workflow = StandardWorkflow(config)
        result = workflow.run(_well())
        return result, next(step for step in result.steps if step.step == "saturation")

    result, step = saturation(saturation_model="auto")
    assert "20 shaly carbonate samples used indonesia instead of Dual Water" in step.decisions[-1]
    assert step.decisions[-1] in result.warnings
    clay, sand = _indonesia_terms(0.05)
    np.testing.assert_allclose(_shaly(result, "SW"), (1 / np.sqrt(10.0)) / (clay + sand), rtol=1e-9)

    result, step = saturation(saturation_model="dual_water")
    assert not step.success and "needs the bound-water resistivity rwb" in step.message
    assert result.well.get("SW") is None

    result, step = saturation(saturation_model="dual_water", rwb=0.02)
    assert step.success and step.parameters["rwb"] == 0.02


# -- Indonesia ---------------------------------------------------------------------


def test_current_workflow_uses_the_poupon_leveaux_indonesia_equation():
    result = _run("current")
    rw = _shaly(result, "RW")
    clay, sand = _indonesia_terms(rw)
    step = _step(result, "saturation")

    assert result.rcl_computed == pytest.approx(2.0)
    assert step.parameters["sw_model_sample_counts"] == {"indonesia": 20}
    np.testing.assert_allclose(_shaly(result, "SW"), (1 / np.sqrt(10.0)) / (clay + sand), rtol=1e-9)
    assert _shaly(result, "SW")[0] == pytest.approx(0.265, abs=1e-3)
    assert step.parameters["shaly_sand_equations"] == "published"
    assert any("Poupon and Leveaux (1971)" in note for note in step.decisions)
    assert not any("Canonical" in note for note in step.decisions)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_keeps_the_sand_term_only_indonesia(calculations):
    result = _run("geolink", calculations=calculations)
    rw = _shaly(result, "RW")
    clay, sand = _indonesia_terms(rw)
    step = _step(result, "saturation")

    np.testing.assert_allclose(_shaly(result, "SW"), (1 / np.sqrt(10.0) - clay) / sand, rtol=1e-9)
    assert _shaly(result, "SW")[0] == pytest.approx(0.021, abs=1e-3)
    assert "Canonical Indonesia and Dual Water equations used" in step.decisions
    assert "shaly_sand_equations" not in step.parameters


def test_steps_saturation_uses_the_published_indonesia_equation():
    well = _well()
    well.add_curve("RW", np.full(well.n_samples, 0.05), "ohm.m", "Formation water resistivity")
    run = Workflow("x", [steps.saturation(model="indonesia", rcl=2.0)]).run(well)
    clay, sand = _indonesia_terms(0.05)
    np.testing.assert_allclose(_shaly(run, "SW"), (1 / np.sqrt(10.0)) / (clay + sand), rtol=1e-9)


_ARCHIE_ONLY_NOTE = "No finite source SW input was available; SW is Archie-computed where inputs are finite"


def test_current_saturation_trace_describes_only_the_models_that_ran():
    indonesia = _step(_run("current"), "saturation").decisions
    assert _ARCHIE_ONLY_NOTE not in indonesia
    assert not any(note.startswith("Dual Water phi_cl") for note in indonesia)

    dual_water = _step(
        _run("current", lithology_domain="carbonate", phi_cl=0.12, rwb=0.02), "saturation"
    ).decisions
    assert _ARCHIE_ONLY_NOTE not in dual_water
    assert "Dual Water phi_cl = 0.120 from config" in dual_water

    clean = _well()
    clean.get("VCL").data[:] = 0.10
    archie = _step(_run("current", clean), "saturation").decisions
    assert _ARCHIE_ONLY_NOTE in archie
    assert not any(note.startswith("Dual Water phi_cl") for note in archie)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_saturation_trace_keeps_its_archie_and_phi_cl_notes(calculations):
    decisions = _step(_run("geolink", calculations=calculations, phi_cl=0.12), "saturation").decisions
    assert _ARCHIE_ONLY_NOTE in decisions
    assert "Dual Water phi_cl = 0.120 from config" in decisions


# -- Dual Water and Rwb --------------------------------------------------------------


def test_current_routes_shaly_carbonate_without_rwb_to_indonesia():
    result = _run("current", lithology_domain="carbonate", phi_cl=0.12)
    rw = _shaly(result, "RW")
    clay, sand = _indonesia_terms(rw)
    step = _step(result, "saturation")

    assert step.parameters["sw_model_sample_counts"] == {"indonesia": 20}
    assert step.parameters["sw_model_fallback_counts"] == {"dual_water->indonesia": 20}
    assert any("Rwb, which is not estimated: 20 shaly carbonate" in note for note in step.decisions)
    np.testing.assert_allclose(_shaly(result, "SW"), (1 / np.sqrt(10.0)) / (clay + sand), rtol=1e-9)


def test_current_runs_dual_water_with_a_supplied_rwb():
    result = _run("current", lithology_domain="carbonate", phi_cl=0.12, rwb=0.02)
    rw = _shaly(result, "RW")
    step = _step(result, "saturation")
    expected = _dual_water_swe(0.25, 10.0, rw, 0.02, 0.35, 0.12)

    assert step.parameters["sw_model_sample_counts"] == {"dual_water": 20}
    np.testing.assert_allclose(_shaly(result, "SW"), expected, rtol=1e-9)
    assert _shaly(result, "SW")[0] == pytest.approx(0.0188, abs=1e-3)


def test_current_sends_saline_shaly_carbonate_without_rwb_to_simandoux():
    # Without Rwb, shaly carbonate takes the clastic routing: Simandoux for
    # water of 20000 ppm or more. 1/Rt = (PHIE^2/Rw) Sw^2 + (VCL/Rcl) Sw, n = 2.
    result = _run("current", lithology_domain="carbonate", phi_cl=0.12, salinity_ppm=35000.0)
    rw = _shaly(result, "RW")
    sand, clay = 0.20 ** 2 / rw, 0.35 / 2.0
    expected = (-clay + np.sqrt(clay ** 2 + 4 * sand / 10.0)) / (2 * sand)
    step = _step(result, "saturation")

    assert step.parameters["sw_model_sample_counts"] == {"simandoux": 20}
    assert step.parameters["sw_model_fallback_counts"] == {"dual_water->simandoux": 20}
    assert any("20 shaly carbonate samples used the Simandoux equation" in note for note in step.decisions)
    np.testing.assert_allclose(_shaly(result, "SW"), expected, rtol=1e-9)
    assert _shaly(result, "SW")[0] == pytest.approx(0.261, abs=1e-3)


def test_current_sends_shaly_carbonate_with_cec_and_b_but_no_rwb_to_waxman_smits():
    result = _run(
        "current", lithology_domain="carbonate", phi_cl=0.12, cec=5.0, b_coeff=3.8, rho_ma=2.65
    )
    rw = _shaly(result, "RW")
    swt = _waxman_smits_n2(0.25, 10.0, rw, 0.05, 3.8)  # 5 meq/100 g = 0.05 meq/g
    step = _step(result, "saturation")

    assert step.parameters["sw_model_sample_counts"] == {"waxman_smits": 20}
    assert step.parameters["sw_model_fallback_counts"] == {"dual_water->waxman_smits": 20}
    np.testing.assert_allclose(_shaly(result, "SW"), 1 - (0.25 / 0.20) * (1 - swt), atol=1e-6)
    assert _shaly(result, "SW")[0] == pytest.approx(0.060, abs=1e-3)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
@pytest.mark.parametrize("inputs", [{}, {"salinity_ppm": 35000.0}, {"cec": 5.0, "b_coeff": 3.8}])
def test_benchmark_keeps_dual_water_with_rwb_from_rw(calculations, inputs):
    result = _run(
        "geolink", calculations=calculations, lithology_domain="carbonate", phi_cl=0.12, **inputs
    )
    rw = _shaly(result, "RW")
    expected = canonical_dual_water(
        np.full(16, 0.25), np.full(16, 10.0), rw, 0.3 * rw, np.full(16, 0.35), 0.12
    )

    assert _step(result, "saturation").parameters["sw_model_sample_counts"] == {"dual_water": 20}
    np.testing.assert_allclose(_shaly(result, "SW"), expected, rtol=1e-12)


def test_explicit_dual_water_needs_rwb():
    result = _run("current", sw_model="dual_water")
    assert any("needs the bound-water resistivity rwb" in error for error in result.errors)
    assert result.well.get("SW") is None
    with pytest.raises(ValueError, match="rwb"):
        _run("current", sw_model="dual_water", raise_on_error=True)
    with pytest.raises(ValueError, match="needs rwb="):
        steps.saturation(model="dual_water")
    # The "geolink" profile keeps Rwb = 0.3 Rw.
    assert _run("geolink", sw_model="dual_water").well.get("SW") is not None


# -- Waxman-Smits: CEC units and Sw basis ----------------------------------------------


def test_current_converts_cec_and_reports_effective_waxman_smits_sw():
    result = _run("current", sw_model="waxman_smits", cec=5.0, b_coeff=3.8, rho_ma=2.65)
    rw = _shaly(result, "RW")
    swt = _waxman_smits_n2(0.25, 10.0, rw, 0.05, 3.8)  # 5 meq/100 g = 0.05 meq/g
    sw = _shaly(result, "SW")
    step = _step(result, "saturation")

    np.testing.assert_allclose(sw, 1 - (0.25 / 0.20) * (1 - swt), atol=1e-6)
    # Hydrocarbon pore volume is the same on both bases.
    np.testing.assert_allclose(0.20 * (1 - sw), 0.25 * (1 - swt), atol=1e-6)
    assert step.parameters["cec_meq_per_g"] == pytest.approx(0.05)
    assert any("5 meq/100 g (0.05 meq/g)" in note for note in step.decisions)


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_passes_cec_unconverted_and_keeps_total_sw(calculations):
    result = _run(
        "geolink", calculations=calculations, sw_model="waxman_smits", cec=5.0, b_coeff=3.8, rho_ma=2.65
    )
    rw = _shaly(result, "RW")
    expected = waxman_smits(np.full(16, 0.25), np.full(16, 10.0), rw, 5.0, 2.65, 3.8)
    np.testing.assert_allclose(_shaly(result, "SW"), expected, rtol=1e-12)


def test_cec_must_be_a_plausible_meq_per_100_g_value():
    with pytest.raises(ValueError, match="cec must be between 0 and 150 meq/100 g"):
        DynamicWorkflowConfig(cec=500.0)
    with pytest.raises(ValueError, match="cec must be at most 150"):
        steps.saturation(model="waxman_smits", cec=500.0, b_coeff=3.8)
    # A value set after the config was made, as LAS header values are, is
    # checked again when the workflow runs.
    config = DynamicWorkflowConfig(rw=0.05, skip_steps=_SATURATION_ONLY)
    config.cec, config.b_coeff = 500.0, 3.8
    with pytest.raises(ValueError, match="cec must be between 0 and 150"):
        DynamicWorkflow(config).run(_well())
    # The "geolink" profile keeps its validation.
    assert DynamicWorkflowConfig(cec=500.0, calculation_profile="geolink").cec == 500.0


def test_current_warns_about_a_cec_that_looks_like_meq_per_g():
    message = "cec=0.05 meq/100 g is a clean-sand value; a meq/g value must be multiplied by 100"
    settings = dict(sw_model="waxman_smits", b_coeff=3.8, rho_ma=2.65)

    assert message in _run("current", cec=0.05, **settings).warnings
    assert not any("clean-sand value" in note for note in _run("current", cec=5.0, **settings).warnings)
    for calculations in BENCHMARK_CALCULATIONS:
        benchmark = _run("geolink", calculations=calculations, cec=0.05, **settings)
        assert not any("clean-sand value" in note for note in benchmark.warnings)
    # steps.saturation warns the same way.
    well = _well()
    well.add_curve("RW", np.full(well.n_samples, 0.05), "ohm.m", "Formation water resistivity")
    run = Workflow("x", [steps.saturation(model="waxman_smits", cec=0.05, b_coeff=3.8)]).run(well)
    assert any(message in note for note in run.warnings)


def test_explicit_waxman_smits_needs_cec_and_b():
    result = _run("current", sw_model="waxman_smits", cec=5.0)
    assert any("needs cec (meq/100 g) and b_coeff" in error for error in result.errors)


def test_steps_saturation_converts_cec_like_the_workflow():
    well = _well()
    well.add_curve("RW", np.full(well.n_samples, 0.05), "ohm.m", "Formation water resistivity")
    step = steps.saturation(model="waxman_smits", cec=5.0, b_coeff=3.8, matrix_density=2.65)
    run = Workflow("x", [step]).run(well)
    swt = _waxman_smits_n2(0.25, 10.0, 0.05, 0.05, 3.8)
    np.testing.assert_allclose(_shaly(run, "SW"), 1 - (0.25 / 0.20) * (1 - swt), atol=1e-6)


def test_effective_waxman_smits_sw_keeps_missing_samples_missing():
    swe = _effective_from_total_sw(
        np.array([np.nan, 0.5, 0.8, 0.5]),
        np.array([0.20, 0.20, 0.25, 0.25]),
        np.array([0.0, 0.0, 0.20, 0.05]),
    )
    # Swt missing stays missing; no effective pore space gives 1;
    # 1 - (0.25/0.20)(1 - 0.8) = 0.75; 1 - (0.25/0.05)(1 - 0.5) < 0 is clipped to 0.
    np.testing.assert_allclose(swe, [np.nan, 1.0, 0.75, 0.0])


def _core_corrected_well() -> Well:
    """``_well()`` with a core-corrected PHIE_CC of 0.16 in the shaly sand (log PHIE 0.20)."""
    well = _well()
    phie_cc = well.get("PHIE").data.copy()
    phie_cc[SHALY] = 0.16
    well.add_curve("PHIE_CC", phie_cc, "v/v", "Core-corrected effective porosity")
    return well


def test_current_waxman_smits_pairs_phie_cc_with_a_total_porosity_on_its_basis():
    result = _run(
        "current", _core_corrected_well(), sw_model="waxman_smits", cec=5.0, b_coeff=3.8, rho_ma=2.65
    )
    rw = _shaly(result, "RW")
    step = _step(result, "saturation")

    # PHIT' = PHIE_CC + (PHIT - PHIE) = 0.16 + 0.05 = 0.21; Swt = 0.293 at Rw 0.05,
    # so SW = 1 - (0.21/0.16)(1 - 0.293) = 0.072.
    swt = _waxman_smits_n2(0.21, 10.0, rw, 0.05, 3.8)
    sw = _shaly(result, "SW")
    np.testing.assert_allclose(sw, 1 - (0.21 / 0.16) * (1 - swt), atol=1e-6)
    np.testing.assert_allclose(0.16 * (1 - sw), 0.21 * (1 - swt), atol=1e-6)
    # The log PHIT with PHIE_CC would read the 0.04 core correction as bound
    # water: 1 - (0.25/0.16)(1 - 0.248) < 0, clipped to SW = 0.
    assert 1 - (0.25 / 0.16) * (1 - _waxman_smits_n2(0.25, 10.0, 0.05, 0.05, 3.8)) < 0
    assert step.parameters["total_porosity_basis"].startswith("PHIE_CC + (PHIT - PHIE)")
    assert step.parameters["waxman_smits_clipped_to_zero"] == 0


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_keeps_the_log_phit_beside_phie_cc(calculations):
    result = _run(
        "geolink", _core_corrected_well(), calculations=calculations,
        sw_model="waxman_smits", cec=5.0, b_coeff=3.8, rho_ma=2.65,
    )
    rw = _shaly(result, "RW")
    expected = waxman_smits(np.full(16, 0.25), np.full(16, 10.0), rw, 5.0, 2.65, 3.8)
    np.testing.assert_allclose(_shaly(result, "SW"), expected, rtol=1e-12)
    parameters = _step(result, "saturation").parameters
    assert "total_porosity_basis" not in parameters
    assert "waxman_smits_clipped_to_zero" not in parameters


def test_effective_waxman_smits_counts_the_samples_clipped_to_zero():
    # Tight shaly sand: PHIE 0.05 of PHIT 0.25, so the bound-water fraction is
    # 0.8, and at Rt 40 ohm.m Swt is 0.109: SW is clipped to 0 and counted.
    well = _well()
    well.get("PHIE").data[SHALY] = 0.05
    well.get("RDEP").data[SHALY] = 40.0
    result = _run("current", well, sw_model="waxman_smits", cec=5.0, b_coeff=3.8, rho_ma=2.65)
    step = _step(result, "saturation")

    swt = _waxman_smits_n2(0.25, 40.0, _shaly(result, "RW"), 0.05, 3.8)
    assert np.all(swt < (0.25 - 0.05) / 0.25)
    np.testing.assert_allclose(_shaly(result, "SW"), 0.0)
    # The shale samples (PHIT 0.16, PHIE 0.02, Rt 2) keep Swt above 0.875.
    assert np.all(result.well.get("SW").data[16:] > 0)
    assert step.parameters["waxman_smits_clipped_to_zero"] == 16
    assert any(
        note.startswith("Waxman-Smits: 16 samples with Swt below the bound-water fraction")
        for note in step.decisions
    )
    module = SaturationModule(model="waxman_smits", effective_waxman_smits=True)
    direct = module.calculate(well, rw=0.05, rcl=2.0, cec=0.05, b_coeff=3.8)
    assert direct.waxman_smits_clipped_to_zero == 16


# -- Oil-based mud --------------------------------------------------------------------


def _obm_well() -> Well:
    well = _well()
    well.get("VCL").data[SHALY] = 0.10
    well.add_curve("RSHA", np.full(well.n_samples, 4.0), "ohm.m", "Shallow resistivity")
    return well


def test_current_writes_no_sxo_or_mhi_for_oil_based_mud():
    skip = [name for name in _SATURATION_ONLY if name not in ("rmf", "mhi")]
    result = _run("current", _obm_well(), mud_type="OBM", skip_steps=skip)
    step = _step(result, "saturation")

    assert result.well.get("SXO") is None
    assert result.well.get("MHI") is None
    assert step.parameters["sxo_method"] is None
    assert step.parameters["sxo_reason"].startswith("mud_type='OBM'")
    assert any(note.startswith("SXO not computed: mud_type='OBM' (") for note in step.decisions)
    assert result.specific_steps_skipped["mhi"].startswith("mud_type='OBM'")


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
def test_benchmark_keeps_sxo_and_mhi_for_oil_based_mud(calculations):
    skip = [name for name in _SATURATION_ONLY if name not in ("rmf", "mhi")]
    result = _run("geolink", _obm_well(), calculations=calculations, mud_type="OBM", skip_steps=skip)

    assert _step(result, "saturation").parameters["sxo_method"] == "archie_rxo_rmf"
    assert result.well.get("SXO") is not None
    assert result.well.get("MHI") is not None


_RW_ONLY = [
    "bad_hole", "vclay", "rcl", "lithology", "invasion", "porosity", "saturation",
    "permeability", "rock_quality", "bvw", "mhi", "net_pay",
]


def _sp_well(sp: bool = True) -> Well:
    """Thirty clean water-bearing samples (SP -60 mV) over ten shale samples (SP 0)."""
    well = Well("OBM-RW")
    well.depth = 2000.0 + 0.5 * np.arange(40)
    well.depth_unit = "m"
    clean = np.arange(40) < 30
    well.add_curve("VCL", np.where(clean, 0.10, 0.90), "v/v", "Clay volume")
    well.add_curve("PHIE", np.where(clean, 0.25, 0.02), "v/v", "Effective porosity")
    well.add_curve("RDEP", np.where(clean, 1.0, 2.0), "ohm.m", "Deep resistivity")
    well.add_curve("RSHA", np.where(clean, 0.8, 2.0), "ohm.m", "Shallow resistivity")
    if sp:
        well.add_curve("SP", np.where(clean, -60.0, 0.0), "mV", "Spontaneous potential")
    return well


def _rw_step(profile, well, *, calculations="main_run", **settings):
    config = DynamicWorkflowConfig(a=1.0, m=2.0, calculation_profile=profile, skip_steps=_RW_ONLY, **settings)
    with _rerun_calculations(calculations == "rerun"):
        return _step(DynamicWorkflow(config).run(well), "rw")


@pytest.mark.parametrize("sp", [True, False])
def test_current_takes_no_rw_from_sp_or_rxo_rt_with_oil_based_mud(sp):
    step = _rw_step("current", _sp_well(sp), mud_type="OBM")

    assert any(note.startswith("Rw SP and Rxo/Rt methods skipped: mud_type='OBM'") for note in step.decisions)
    # The clean-wet Rwa sets Rw instead: RDEP * PHIE^m / a = 1.0 * 0.25^2.
    assert step.method == "rwa_clean_wet_p5"
    assert step.parameters["rw_from_rwa"] == pytest.approx(0.0625)
    # With water-based mud the SP, or else Rxo/Rt, sets Rw.
    wbm = _rw_step("current", _sp_well(sp), rmf_at_meas_temp=0.5)
    assert wbm.method == ("sp_clean_water" if sp else "rxo_rt_clean_water")


@pytest.mark.parametrize("calculations", BENCHMARK_CALCULATIONS)
@pytest.mark.parametrize("sp", [True, False])
def test_benchmark_keeps_the_rw_from_the_oil_based_mud_rmf(calculations, sp):
    step = _rw_step("geolink", _sp_well(sp), calculations=calculations, mud_type="OBM")

    assert step.method == ("sp_clean_water" if sp else "rxo_rt_clean_water")
    assert step.parameters["rmf_min"] == 200.0


# -- Routing threshold -------------------------------------------------------------------


def test_routing_threshold_is_0_3_unless_set():
    assert _step(_run("current"), "saturation").parameters["vcl_cutoff_for_routing"] == 0.3

    well = _well()
    well.get("VCL").data[SHALY] = 0.20
    result = _run("current", well, sw_routing_vcl_cutoff=0.15)
    step = _step(result, "saturation")

    assert step.parameters["vcl_cutoff_for_routing"] == 0.15
    assert step.parameters["sw_model_sample_counts"] == {"indonesia": 20}
    # Net pay resolves its unset cutoffs per lithology family (clastic VCL < 0.40).
    assert _step(result, "net_pay").parameters["vcl_cutoff"] == 0.4
    with pytest.raises(ValueError, match="sw_routing_vcl_cutoff"):
        DynamicWorkflowConfig(sw_routing_vcl_cutoff=1.5)


# -- Context export text -------------------------------------------------------------------


def test_context_text_names_the_equation_each_profile_ran():
    current = {"shaly_sand_equations": "published", "rwb": None, "vcl_cutoff_for_routing": 0.3}
    published = reference._sw_equation_text(["indonesia", "waxman_smits"], current)
    assert "Indonesia (Poupon-Leveaux) 1/sqrt(Rt)=[VCL^(1-VCL/2)/sqrt(RCL)+PHIE^(m/2)/sqrt(a*RW)]*Sw^(n/2)" in published
    assert "Sw=1-(PHIT/PHIE)*(1-Swt)" in published
    assert reference._sw_route_reason_text("indonesia", current).endswith(
        "; non-carbonate, or carbonate without Rwb"
    )
    # The text of the "geolink" profile is the eval package's.
    assert reference._sw_equation_text(["indonesia"], {}) == (
        "Indonesia 1/sqrt(Rt)=VCL^(1-VCL/2)/sqrt(RCL)+PHIE^(m/2)*Sw^(n/2)/sqrt(a*RW)"
    )
    assert reference._sw_equation_text(["dual_water"], {"rwb": None}).startswith(
        "Dual Water phi_bw=VCL*phi_cl, Swb=phi_bw/PHIT, 1/Rwt=Swb/Rwb+(1-Swb)/RW, "
        "Sw=[a*Rwt/(PHIT^m*Rt)]^(1/n), Rwb=0.3*RW"
    )


def test_context_route_text_names_the_route_that_reached_indonesia():
    current = {
        "shaly_sand_equations": "published", "rwb": None, "vcl_cutoff_for_routing": 0.3,
        "sw_model_requested": "auto",
    }
    # Shaly carbonate without Rwb takes the clastic routing, so it reaches
    # Indonesia on the same inputs as a non-carbonate sample.
    assert reference._sw_route_reason_text("indonesia", current) == (
        "VCL>=0.3, no CEC/B, no salinity>=20000 ppm; non-carbonate, or carbonate without Rwb"
    )
    # With an Rwb, shaly carbonate goes to Dual Water.
    assert reference._sw_route_reason_text("indonesia", {**current, "rwb": 0.02}) == (
        "VCL>=0.3, non-carbonate, no CEC/B, no salinity>=20000 ppm"
    )
    # A requested model runs on every sample, whatever its VCL.
    requested = {**current, "sw_model_requested": "indonesia"}
    assert reference._sw_route_reason_text("indonesia", requested) == "requested sw_model='indonesia'"
    # The text of the "geolink" profile is the eval package's.
    benchmark = {"vcl_cutoff_for_routing": 0.3, "salinity_ppm": 35000.0, "sw_model_requested": "indonesia"}
    assert reference._sw_route_reason_text("indonesia", benchmark) == (
        "VCL>=0.3, non-carbonate, no CEC/B, salinity=35000 ppm"
    )


def test_context_routing_counts_follow_the_rwb_rule_of_the_current_profile():
    well = _well()
    well.add_curve("LITHOLOGY_GEOLINK", np.full(well.n_samples, 11.0), "", "Limestone")
    well.add_curve("SW", np.full(well.n_samples, 0.5), "v/v", "Water saturation")
    benchmark = {"sw_model_requested": "auto"}
    current = {**benchmark, "shaly_sand_equations": "published", "rwb": None}

    assert reference._segment_saturation_model_counts(well, benchmark) == {"dual_water": 20}
    assert reference._segment_saturation_model_counts(well, current) == {"indonesia": 20}
    assert reference._segment_saturation_model_counts(well, {**current, "rwb": 0.02}) == {"dual_water": 20}
    # Without Rwb, shaly carbonate takes the clastic routing.
    saline, lab = {"salinity_ppm": 35000.0}, {"cec": 5.0, "b_coeff": 3.8}
    assert reference._segment_saturation_model_counts(well, {**current, **saline}) == {"simandoux": 20}
    assert reference._segment_saturation_model_counts(well, {**current, **lab}) == {"waxman_smits": 20}
    # The "geolink" profile keeps Dual Water for shaly carbonate.
    assert reference._segment_saturation_model_counts(well, {**benchmark, **saline}) == {"dual_water": 20}
    assert reference._segment_saturation_model_counts(well, {**benchmark, **lab}) == {"dual_water": 20}


# -- The pp.interpret entry point and LAS header values ------------------------------------


@pytest.mark.parametrize("profile", ["current", "geolink"])
def test_interpret_indonesia_matches_the_equation_of_its_profile(profile):
    result = pp.interpret(
        _well(), rw=0.05, a=1.0, m=2.0, n=2.0, sw_model="indonesia",
        calculation_profile=profile, skip_steps=_SATURATION_ONLY,
    )
    well = result.well
    equation = indonesia if profile == "current" else canonical_indonesia
    expected = equation(
        well.get("RDEP").data, well.get("RW").data, well.get("PHIE").data,
        well.get("VCL").data, result.rcl_computed,
    )
    np.testing.assert_allclose(well.get("SW").data, expected, rtol=1e-12)


def test_interpret_reads_a_las_header_cec_in_meq_per_100_g():
    def run(profile):
        well = _well()
        well.set_parameter("CEC", 5.0)
        well.set_parameter("B_COEFF", 3.8)
        return pp.interpret(
            well, rw=0.05, a=1.0, m=2.0, n=2.0, rho_ma=2.65, sw_model="waxman_smits",
            calculation_profile=profile, skip_steps=_SATURATION_ONLY,
        )

    current = run("current")
    rw = _shaly(current, "RW")
    swt = _waxman_smits_n2(0.25, 10.0, rw, 0.05, 3.8)  # 5 meq/100 g = 0.05 meq/g
    assert _step(current, "saturation").parameters["cec"] == 5.0
    np.testing.assert_allclose(_shaly(current, "SW"), 1 - (0.25 / 0.20) * (1 - swt), atol=1e-6)
    # The "geolink" profile passes the header value to the solver as it stands.
    benchmark = run("geolink")
    expected = waxman_smits(np.full(16, 0.25), np.full(16, 10.0), rw, 5.0, 2.65, 3.8)
    np.testing.assert_allclose(_shaly(benchmark, "SW"), expected, rtol=1e-12)


@pytest.mark.parametrize("unit, cec", [("MEQ/G", 5.0), ("meq/g", 5.0), ("MEQ/100G", 0.05), ("", 0.05)])
def test_a_las_header_cec_labelled_meq_per_g_is_converted_under_current(unit, cec):
    well = _well()
    well.set_parameter("CEC", 0.05)
    if unit:
        well.set_parameter("CEC__UNIT", unit)

    assert _extract_global_parameter_overrides(well)[0]["cec"] == pytest.approx(cec)
    overrides, _ = _extract_global_parameter_overrides(well, calculation_profile="geolink")
    assert overrides["cec"] == 0.05


def test_interpret_reads_a_las_header_cec_in_meq_per_g():
    well = _well()
    well.set_parameter("CEC", 0.05)
    well.set_parameter("CEC__UNIT", "MEQ/G")
    well.set_parameter("B_COEFF", 3.8)
    result = pp.interpret(
        well, rw=0.05, a=1.0, m=2.0, n=2.0, rho_ma=2.65, sw_model="waxman_smits",
        skip_steps=_SATURATION_ONLY,
    )
    swt = _waxman_smits_n2(0.25, 10.0, _shaly(result, "RW"), 0.05, 3.8)

    assert _step(result, "saturation").parameters["cec"] == pytest.approx(5.0)
    np.testing.assert_allclose(_shaly(result, "SW"), 1 - (0.25 / 0.20) * (1 - swt), atol=1e-6)
    assert not any("clean-sand value" in note for note in result.warnings)
