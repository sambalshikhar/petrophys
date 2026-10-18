"""Tests for the decided workflow changes (release_checks/METHOD_DECISIONS.md)."""

import dataclasses
import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import DynamicWorkflowConfig
from petrophys.workflow.standard_workflow import run_standard_workflow
from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones


def _well(name="30_6-5"):
    return pp.read(pp.sample(name))


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _drop_header_rw(well):
    for key in ("RW", "RWT"):
        well._parameters.pop(key, None)
    return well


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --- Rmf precedence and measurement temperatures ---------------------------------


def test_explicit_rm_wins_over_header_rmf():
    # 30/6-5 carries RMF 0.08 ohm.m in its header.
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, rm=0.2, rm_meas_temp_f=60)
    rmf = _step(result, "rmf")
    assert rmf.method == "rm_to_rmf_conversion_factor_0.75"
    assert rmf.parameters["source_priority"] == "config.rm"

    historical = pp.interpret(
        _well(), rw=0.07, rw_meas_temp_f=230, rm=0.2, rm_meas_temp_f=60, calculation_profile="geolink"
    )
    assert _step(historical, "rmf").method.startswith("las_metadata")


def test_explicit_chlorides_and_salinity_win_over_header_rmf():
    chlorides = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, rmf_chlorides_mg_l=30000)
    assert _step(chlorides, "rmf").method == "mud_filtrate_chlorides_conversion"
    salinity = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, rmf_salinity_ppm=40000)
    assert _step(salinity, "rmf").method == "mud_filtrate_salinity_conversion"


def test_header_clfl_ranks_below_header_rmf():
    # pp.interpret copies the header CLFL into rmf_chlorides_mg_l; it is not an explicit setting.
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230)
    rmf = _step(result, "rmf")
    assert rmf.method.startswith("las_metadata")
    assert result.config.rmf_chlorides_mg_l == pytest.approx(20366.0)
    assert any("Header CLFL" in d and "not used" in d for d in rmf.decisions)


def test_explicit_default_measurement_temperature_is_used():
    well = _well()
    well.set_parameter("RMFT", 100.0)
    unset = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    assert _step(unset, "rmf").parameters["measured_temp_f"] == pytest.approx(100.0)
    explicit = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, rmf_meas_temp_f=75.0)
    assert _step(explicit, "rmf").parameters["measured_temp_f"] == pytest.approx(75.0)
    assert DynamicWorkflowConfig().rmf_meas_temp_f is None
    assert DynamicWorkflowConfig().rm_meas_temp_f is None


def test_explicit_default_phi_cl_counts_as_configured():
    explicit = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, phi_cl=0.15)
    porosity = _step(explicit, "porosity").parameters
    assert porosity["phi_cl"] == pytest.approx(0.15)
    assert porosity["phi_cl_source"] == "config"
    unset = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230)
    assert _step(unset, "porosity").parameters["phi_cl_source"] != "config"
    assert DynamicWorkflowConfig().phi_cl is None


# --- Zone-only header parameters ---------------------------------------------------


def _zoned_well(value=0.3, fraction=(0.4, 0.6)):
    well = _drop_header_rw(_well())
    depth = np.asarray(well.depth, dtype=float)
    top = float(np.nanmin(depth) + fraction[0] * (np.nanmax(depth) - np.nanmin(depth)))
    bottom = float(np.nanmin(depth) + fraction[1] * (np.nanmax(depth) - np.nanmin(depth)))
    well.set_parameter("TOP_Z1", top)
    well.set_parameter("BOT_Z1", bottom)
    well.set_parameter("RW_Z1", value)
    return well, (depth >= top) & (depth <= bottom)


def test_zone_only_rw_applies_inside_its_zone_only():
    well, inside = _zoned_well()
    zoned = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    plain = pp.interpret(_drop_header_rw(_well()), rw=0.07, rw_meas_temp_f=230)
    zoned_rw = zoned.well["RW"].data
    plain_rw = plain.well["RW"].data
    np.testing.assert_allclose(zoned_rw[~inside], plain_rw[~inside])
    assert np.all(zoned_rw[inside] > plain_rw[inside])
    rw = _step(zoned, "rw")
    assert rw.parameters["zone_only_rw"][0]["parameter"] == "RW_Z1"
    assert any("RW_Z1" in w and "only" in w for w in zoned.warnings)


def test_zone_covering_the_whole_well_is_a_constant_and_geolink_keeps_the_old_reading():
    well, _ = _zoned_well(fraction=(0.0, 1.0))
    result = pp.interpret(well)
    assert _step(result, "rw").parameters["source_priority"] == "las.RW"
    assert "zone_only_rw" not in _step(result, "rw").parameters

    well, _ = _zoned_well()
    historical = pp.interpret(well, calculation_profile="geolink")
    rw = _step(historical, "rw").parameters
    assert rw["source_priority"] == "las.RW" and rw["rw"] == pytest.approx(0.3)
    assert "zone_only_rw" not in rw


def test_zoned_runner_does_not_fill_uncovered_intervals_from_zone_values():
    well, _ = _zoned_well()
    current = run_dynamic_workflow_with_zones(well, DynamicWorkflowConfig())
    assert current.base_result.config.rw is None
    assert any("inside their zones only" in w for w in current.warnings)

    well, _ = _zoned_well()
    historical = run_dynamic_workflow_with_zones(well, DynamicWorkflowConfig(calculation_profile="geolink"))
    assert historical.base_result.config.rw == pytest.approx(0.3)


# --- Hydrocarbon correction, temperature log ---------------------------------------


def test_hc_correction_feeds_the_porosity_step():
    well = _well()
    well.add_curve("SXO", np.full(well.n_samples, 0.5), "v/v", "Flushed zone saturation")
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, apply_hc_correction=True)
    porosity = _step(result, "porosity")
    assert {"NPHI_HC", "RHOB_HC"} <= set(porosity.input_curves)
    plain = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230)
    assert not np.allclose(result.well["PHID"].data, plain.well["PHID"].data, equal_nan=True)

    historical = pp.interpret(
        well, rw=0.07, rw_meas_temp_f=230, apply_hc_correction=True, calculation_profile="geolink"
    )
    assert "NPHI" in _step(historical, "porosity").input_curves


def test_measured_temperature_log_is_the_source_and_gaps_are_modelled():
    well = _well()
    measured = np.linspace(10.0, 120.0, well.n_samples)
    gaps = np.zeros(well.n_samples, dtype=bool)
    gaps[::4] = True  # 25% missing
    measured[gaps] = np.nan
    well.add_curve("TEMP", measured, "degC", "Measured")
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    temp = result.well["TEMP"].data
    step = _step(result, "temperature")
    assert step.method == "measured_temperature_log"
    np.testing.assert_allclose(temp[~gaps], measured[~gaps] * 9 / 5 + 32)
    assert np.isfinite(temp[gaps]).all()
    assert step.parameters["temp_log_gap_samples"] == int(gaps.sum())
    assert result.well.get("TEMP_LOG") is not None

    sparse = _well()
    sparse_log = np.full(sparse.n_samples, np.nan)
    sparse_log[: sparse.n_samples // 4] = 50.0
    sparse.add_curve("TEMP", sparse_log, "degC", "Measured")
    replaced = pp.interpret(sparse, rw=0.07, rw_meas_temp_f=230)
    assert _step(replaced, "temperature").method != "measured_temperature_log"
    assert any("was replaced" in w for w in replaced.warnings)


# --- Core-corrected curves ---------------------------------------------------------


def test_perm_cc_is_limited_to_the_plug_porosity_range():
    result = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230)
    core = _step(result, "core_correction")
    outside = core.parameters["perm_cc_outside_plug_range_samples"]
    assert outside > 0
    (fit,) = core.parameters["perm_cc_plug_porosity_ranges"]
    assert 0 < fit["plug_porosity_min"] < fit["plug_porosity_max"] < 0.5
    perm_cc = result.well["PERM_CC"].data
    phie_cc = result.well["PHIE_CC"].data
    assert np.count_nonzero(np.isnan(perm_cc) & np.isfinite(phie_cc)) >= outside

    historical = pp.interpret(_well(), rw=0.07, rw_meas_temp_f=230, calculation_profile="geolink")
    assert "perm_cc_outside_plug_range_samples" not in _step(historical, "core_correction").parameters
    assert np.count_nonzero(np.isfinite(historical.well["PERM_CC"].data)) > np.count_nonzero(np.isfinite(perm_cc))


def test_input_core_curves_are_kept_as_input_copies():
    original = _well("HW-3")
    phie_cc = np.array(original["PHIE_CC"].data, copy=True)
    result = pp.interpret(original)
    np.testing.assert_allclose(result.well["PHIE_CC_INPUT"].data, phie_cc, equal_nan=True)
    assert result.well.get("PERM_CC_INPUT") is not None
    assert any("kept as PHIE_CC_INPUT" in w for w in result.warnings)

    historical = pp.interpret(_well("HW-3"), calculation_profile="geolink")
    assert historical.well.get("PHIE_CC_INPUT") is None

    standard = run_standard_workflow(_well("HW-3"), bht_f=251.0, td_ft=8299.0)
    assert standard.well.get("PHIE_CC_INPUT") is not None
    assert any("kept as PHIE_CC_INPUT" in w for w in standard.warnings)


# --- SP-derived Rw -------------------------------------------------------------------


def _sp_well():
    well = _drop_header_rw(_well())
    return well


def test_sp_rw_is_rejected_when_half_the_values_lie_outside_the_range(monkeypatch):
    import petrophys.interpretation.electrical.rw.sp_rw as sp_rw

    def outside(sp_clean, sp_shale, rmf, temperature, temp_unit="F"):
        return np.full(np.shape(temperature), 0.001)

    monkeypatch.setattr(sp_rw, "rw_from_sp", outside)
    rejected = pp.interpret(_sp_well())
    assert _step(rejected, "rw").method != "sp_clean_water"
    assert any("outside the 0.01-10 ohm.m range" in w for w in rejected.warnings)

    def inside(sp_clean, sp_shale, rmf, temperature, temp_unit="F"):
        return np.full(np.shape(temperature), 0.05)

    monkeypatch.setattr(sp_rw, "rw_from_sp", inside)
    accepted = pp.interpret(_sp_well())
    assert _step(accepted, "rw").method == "sp_clean_water"


# --- Config alias and standard workflow ----------------------------------------------


def test_lithology_alias_fills_an_unset_domain_and_the_domain_wins():
    alias = DynamicWorkflowConfig(lithology="carbonate")
    assert alias.lithology_domain == "carbonate"
    domain = DynamicWorkflowConfig(lithology_domain="carbonate")
    assert domain.lithology == "carbonate"  # the deprecated alias mirrors the domain
    assert dataclasses.replace(domain, lithology_domain="clastic").lithology_domain == "clastic"
    assert DynamicWorkflowConfig().lithology_domain == "clastic"
    # When both are given the domain wins (replace() passes the old mirrored alias along).
    assert DynamicWorkflowConfig(lithology="carbonate", lithology_domain="clastic").lithology_domain == "clastic"
    assert dataclasses.replace(alias, lithology_domain="clastic").lithology == "clastic"


def test_standard_workflow_endpoints_follow_matrix_type_unless_explicit():
    options = dict(calculate_sw=False, calculate_perm=False, calculate_net_pay=False)
    limestone = run_standard_workflow(_well("HW-3"), matrix_type="limestone", **options)
    porosity = next(step for step in limestone.steps if step.step == "porosity")
    assert porosity.parameters["matrix_density"] == pytest.approx(2.71)
    assert porosity.parameters["matrix_dt"] == pytest.approx(47.6)
    assert porosity.parameters["fluid_dt"] == pytest.approx(189.0)

    explicit = run_standard_workflow(
        _well("HW-3"), matrix_type="limestone", matrix_density=2.65, matrix_dt=55.5, **options
    )
    porosity = next(step for step in explicit.steps if step.step == "porosity")
    assert porosity.parameters["matrix_density"] == pytest.approx(2.65)
    assert porosity.parameters["matrix_dt"] == pytest.approx(55.5)
    dt = (explicit.well.get("DTC") or explicit.well.get("DT")).data
    expected = np.clip((dt - 55.5) / (189.0 - 55.5), 0, 0.5)
    finite = np.isfinite(dt)
    np.testing.assert_allclose(explicit.well["PHIS"].data[finite], expected[finite])
