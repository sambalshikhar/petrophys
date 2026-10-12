"""Tests for auto lithology/matrix resolution (vendor log -> solver -> default)."""

import numpy as np
import pytest

from petrophys.model import Well
from petrophys.interpretation.lithology.matrix_inference import (
    LITHOLOGY_CODE_CLASS,
    resolve_from_lithology_log,
    resolve_from_minerals,
    resolve_well_lithology,
)


def _well_with_lithology_codes(codes):
    w = Well("LITH")
    w.depth = np.arange(len(codes), dtype=float)
    w.add_curve("LITHOLOGY_GEOLINK", np.asarray(codes, dtype=float), "_", "vendor lithology")
    return w


def _well_with_minerals(vq, vc, vd):
    n = len(vq)
    w = Well("MIN")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("VQUAR", np.asarray(vq, dtype=float), "v/v", "quartz")
    w.add_curve("VCALC", np.asarray(vc, dtype=float), "v/v", "calcite")
    w.add_curve("VDOLO", np.asarray(vd, dtype=float), "v/v", "dolomite")
    return w


def test_code_class_map_basics():
    assert LITHOLOGY_CODE_CLASS[1] == "clastic"     # sandstone
    assert LITHOLOGY_CODE_CLASS[8] == "clastic"     # shale
    assert LITHOLOGY_CODE_CLASS[9] == "carbonate"   # chalk
    assert LITHOLOGY_CODE_CLASS[11] == "carbonate"  # limestone
    assert LITHOLOGY_CODE_CLASS[15] == "dolomite"   # dolomite


def test_vendor_log_clastic_dominant():
    # 80 clastic (shale=8) + 20 limestone (=11)
    codes = [8] * 80 + [11] * 20
    res = resolve_from_lithology_log(_well_with_lithology_codes(codes), warn=False)
    assert res.lithology == "clastic"
    assert res.matrix == "sandstone"
    assert res.method == "vendor_log"
    assert res.confidence == pytest.approx(0.8, abs=1e-6)


def test_vendor_log_carbonate_dominant():
    codes = [11] * 70 + [8] * 30      # limestone-dominant
    res = resolve_from_lithology_log(_well_with_lithology_codes(codes), warn=False)
    assert res.lithology == "carbonate"
    assert res.matrix == "limestone"


def test_vendor_log_dolomite_picks_dolomite_matrix():
    codes = [15] * 60 + [8] * 30      # dolomite-dominant carbonate
    res = resolve_from_lithology_log(_well_with_lithology_codes(codes), warn=False)
    assert res.lithology == "carbonate"
    assert res.matrix == "dolomite"


def test_vendor_log_detection_is_robust_to_curve_name():
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_lithology_log
    codes = [8] * 60 + [11] * 20   # clastic-dominant code log
    for name in ("ROCK_TYPE", "FACIES", "LITHFAC", "Lithology_Interp", "GEOL_LITH"):
        w = Well("S")
        w.depth = np.arange(len(codes), dtype=float)
        w.add_curve(name, np.asarray(codes, dtype=float), "_", "lith")
        res = resolve_from_lithology_log(w, warn=False)
        assert res is not None and res.method == "vendor_log", name
        assert res.lithology == "clastic"


def test_vendor_log_detection_rejects_non_code_curves():
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_lithology_log
    # continuous "lithostatic" curve must NOT be treated as a lithology log
    w = Well("C")
    w.depth = np.arange(80, dtype=float)
    w.add_curve("LITHOSTATIC", np.linspace(2000.0, 2500.0, 80), "psi", "p")
    assert resolve_from_lithology_log(w, warn=False) is None


def test_vendor_log_detection_excludes_computed_litho():
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_lithology_log
    # the workflow's own computed "LITHO" (different code scheme) must be ignored
    w = Well("K")
    w.depth = np.arange(80, dtype=float)
    w.add_curve("LITHO", np.asarray([1, 2, 3, 4] * 20, dtype=float), "", "computed")
    assert resolve_from_lithology_log(w, warn=False) is None


def test_vendor_log_absent_returns_none():
    w = Well("NOLITH")
    w.depth = np.arange(50, dtype=float)
    w.add_curve("GR", np.full(50, 40.0), "gAPI", "gr")
    assert resolve_from_lithology_log(w, warn=False) is None


def test_minerals_clastic_vs_carbonate():
    n = 50
    clastic = resolve_from_minerals(
        _well_with_minerals([0.6] * n, [0.05] * n, [0.02] * n), warn=False)
    assert clastic.lithology == "clastic" and clastic.matrix == "sandstone"
    carb = resolve_from_minerals(
        _well_with_minerals([0.05] * n, [0.6] * n, [0.02] * n), warn=False)
    assert carb.lithology == "carbonate" and carb.matrix == "limestone"


def test_resolver_prefers_vendor_log_over_minerals():
    # vendor log says clastic; minerals say carbonate -> vendor log wins
    n = 60
    w = _well_with_minerals([0.05] * n, [0.6] * n, [0.02] * n)
    w.add_curve("LITHOLOGY_GEOLINK", np.full(n, 8.0), "_", "vendor (shale)")
    res = resolve_well_lithology(w, warn=False)
    assert res.method == "vendor_log"
    assert res.lithology == "clastic"


def _well_with_core_porosity(rho_matrix, phi=0.20, n=60):
    w = Well("CORE")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("GR", np.full(n, 20.0), "gAPI", "clean")
    rhob = rho_matrix * (1 - phi) + 1.0 * phi
    w.add_curve("RHOB", np.full(n, rhob), "g/cm3", "Bulk Density")
    w.add_curve("PHIE_CC", np.full(n, phi), "v/v", "core porosity")
    return w


def test_core_matrix_density_identifies_carbonate_without_pef():
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_core_matrix
    # apparent matrix density from core porosity should recover the matrix
    r_ls = resolve_from_core_matrix(_well_with_core_porosity(2.71), warn=False)
    assert r_ls.lithology == "carbonate" and r_ls.matrix == "limestone"
    r_ss = resolve_from_core_matrix(_well_with_core_porosity(2.65), warn=False)
    assert r_ss.lithology == "clastic" and r_ss.matrix == "sandstone"
    r_dol = resolve_from_core_matrix(_well_with_core_porosity(2.87), warn=False)
    assert r_dol.lithology == "carbonate" and r_dol.matrix == "dolomite"


def test_core_matrix_hc_correction_recovers_carbonate():
    # A hydrocarbon-bearing limestone with no water leg: assuming water (Sw=1)
    # biases apparent matrix density low (reads quartz); the Sw-based HC fluid
    # correction recovers carbonate.
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_core_matrix
    n, phi = 80, 0.20
    # Build RHOB for a limestone whose flushed zone retains residual oil:
    # rho_fl_eff at Sw=0.2 with oil 0.8 -> 0.2 + 0.8*0.8 = 0.84
    rho_fl_true = 0.84
    rhob = 2.71 * (1 - phi) + rho_fl_true * phi
    w = Well("HC-LS")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("GR", np.full(n, 20.0), "gAPI", "clean")
    w.add_curve("RHOB", np.full(n, rhob), "g/cm3", "Bulk Density")
    w.add_curve("PHIE_CC", np.full(n, phi), "v/v", "core porosity")
    w.add_curve("SW", np.full(n, 0.20), "v/v", "Water Saturation")

    # water assumption mislabels it clastic
    assert resolve_from_core_matrix(
        w, rho_hc=1.0, warn=False).lithology == "clastic"
    # HC-corrected (oil) recovers carbonate
    assert resolve_from_core_matrix(
        w, rho_hc=0.8, warn=False).lithology == "carbonate"


def test_core_matrix_absent_returns_none():
    from petrophys.interpretation.lithology.matrix_inference import resolve_from_core_matrix
    w = Well("NOCORE")
    w.depth = np.arange(40, dtype=float)
    w.add_curve("RHOB", np.full(40, 2.4), "g/cm3", "d")  # no core porosity
    assert resolve_from_core_matrix(w, warn=False) is None


def test_resolver_prefers_core_matrix_over_solver():
    # core matrix says carbonate; resolve_well_lithology should use it (ranked
    # above the solver), method == core_matrix_density
    from petrophys.interpretation.lithology.matrix_inference import resolve_well_lithology
    res = resolve_well_lithology(_well_with_core_porosity(2.71), warn=False)
    assert res.method == "core_matrix_density"
    assert res.lithology == "carbonate"


def test_resolver_default_when_nothing_available():
    w = Well("EMPTY")
    w.depth = np.arange(30, dtype=float)
    res = resolve_well_lithology(w, warn=False)
    assert res.method == "default"
    assert res.lithology == "clastic" and res.matrix == "sandstone"


def _runnable_well(n=200):
    """A minimal Well with the curves the dynamic workflow needs (no vendor log)."""
    rng = np.random.RandomState(1)
    w = Well("AUTO-TEST")
    w.depth = np.arange(8000.0, 8000.0 + n * 0.5, 0.5)[:n]
    w.add_curve("GR", rng.uniform(20, 120, n), "gAPI", "Gamma Ray")
    w.add_curve("RHOB", rng.uniform(2.2, 2.7, n), "g/cm3", "Bulk Density")
    w.add_curve("NPHI", rng.uniform(0.05, 0.35, n), "v/v", "Neutron Porosity")
    w.add_curve("RDEP", rng.uniform(1, 100, n), "ohm.m", "Deep Resistivity")
    return w


def test_workflow_auto_resolves_and_does_not_mutate_caller_config():
    # No vendor lithology log -> tier-1 solver convergence resolves "auto".
    from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

    well = _runnable_well()
    cfg = DynamicWorkflowConfig(
        rw=0.05, bht_f=200.0, td_ft=10000.0,
        matrix_type="auto", lithology="auto", sw_model="auto",
    )
    result = run_dynamic_workflow(well, cfg)

    # caller's config object is never mutated by auto-resolution
    assert cfg.matrix_type == "auto"
    assert cfg.lithology == "auto"

    # the per-run config copy is resolved to concrete values
    assert result.config.matrix_type in {"sandstone", "limestone", "dolomite"}
    assert result.config.lithology in {"clastic", "carbonate"}
    assert any("Auto lithology resolved" in w for w in result.warnings)


def test_zero_and_flagged_minerals_do_not_vote_for_a_matrix():
    well = _well_with_minerals([0.0] * 30, [0.0] * 30, [0.0] * 30)
    assert resolve_from_minerals(well, warn=False) is None
    well['VCALC'].data[:] = 0.8
    well.add_curve('MMQC', np.ones(30), '', 'Poor fit')
    assert resolve_from_minerals(well, warn=False) is None


def test_module_selects_carbonate_from_core_without_well_name_rules():
    from petrophys.interpretation.lithology import LithologyModule

    well = _well_with_core_porosity(2.71)
    well.name = 'arbitrary-user-well'
    well.add_curve('NPHI', np.full(60, 0.2), 'v/v')
    well.add_curve('VCL', np.zeros(60), 'v/v')
    result = LithologyModule().calculate(well)
    assert result.mineral_selection['source'] == 'core_matrix_density'
    assert result.mineral_selection['reduced_minerals'] == ['calcite', 'dolomite']
    assert np.all(result.solved)
    np.testing.assert_allclose(result.v_calcite, 0.8, atol=1e-7)


@pytest.mark.parametrize('profile, method, illite', [
    ('benchmark', 'costa_gr_illite_calcite_dolomite', True),
    ('current', 'generic_calcite_dolomite_density_closure', False),
])
def test_explicit_carbonate_setting_passes_through_workflow(profile, method, illite):
    from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig

    well = _well_with_core_porosity(2.71)
    well.remove_curve('PHIE_CC')
    well.add_curve('NPHI', np.full(60, 0.2), 'v/v')
    result = DynamicWorkflow(DynamicWorkflowConfig(
        lithology_domain='carbonate', matrix_type='limestone', calculation_profile=profile,
    )).run(well)
    step = next(step for step in result.steps_executed if step.step == 'lithology')
    assert step.method == method
    assert step.parameters['lithology_domain'] == 'carbonate'
    assert ('VILL' in step.output_curves) is illite


def test_custom_mineral_set_is_preserved_for_a_carbonate_well():
    from petrophys.interpretation.lithology import LithologyModule

    well = _well_with_core_porosity(2.71)
    well.add_curve('NPHI', np.full(60, 0.2), 'v/v')
    result = LithologyModule(minerals=['quartz', 'calcite']).calculate(well)
    assert result.mineral_selection['source'] == 'custom_solver'
    assert result.mineral_selection['reduced_minerals'] == ['quartz', 'calcite']


@pytest.mark.parametrize('profile, method', [
    ('benchmark', 'costa_gr_illite_calcite_dolomite'),
    ('current', 'generic_calcite_dolomite_density_closure'),
])
def test_auto_resolution_selects_the_carbonate_domain_route(profile, method):
    from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig

    well = _well_with_core_porosity(2.71)
    well['RHOB'].data[-20:] = 2.65 * 0.8 + 0.2
    well.add_curve('NPHI', np.full(60, 0.2), 'v/v')
    result = DynamicWorkflow(DynamicWorkflowConfig(
        lithology_domain='auto', matrix_type='auto', calculation_profile=profile,
    )).run(well)
    step = next(step for step in result.steps_executed if step.step == 'lithology')
    assert result.config.lithology_domain == 'carbonate'
    assert step.method == method
