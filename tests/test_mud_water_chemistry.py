"""Mud and water chemistry and temperature units, checked against hand-computed values.

Rmf from mud-filtrate chlorides: Cl x 58.44/35.45 is the NaCl equivalent in
mg/l, divided by the brine density at standard conditions (McCain, 1991) to
give ppm by weight; Rmf75 = 0.0123 + 3647.5 / ppm^0.955 (Bateman and Konen,
1977); Arps (1953) carries a resistivity to formation temperature,
R(T) = R(T1) (T1 + 6.77) / (T + 6.77) in degF. The "geolink" profile keeps
reading the CLFL header as NaCl ppm, ignores MFST and MST, and reads TEMP as degF.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.interpretation.electrical.rmf import RmfModule
from petrophys.interpretation.electrical.rw import RwModule, rw_from_salinity, salinity_from_rw
from petrophys.interpretation.electrical.rw.salinity_rw import (
    NACL_PER_CHLORIDE,
    brine_density,
    chlorides_to_nacl_ppm,
)
from petrophys.interpretation.electrical.rw.sp_rw import rw_from_ssp
from petrophys.interpretation.electrical.rw.temp_correction import rw_at_reference_temperature
from petrophys.interpretation.temperature import (
    TemperatureProfile,
    arps_equation,
    calculate_temperature_gradient,
    estimate_gradient_from_bht,
)
from petrophys.model.units import UnitConverter, temperature_unit
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.preprocessing.standardization.unit_converter import convert_units
from petrophys.reporting.reference_export import _rmf_curve_note
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import WorkflowStep
from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

# 30,000 mg/l chlorides is 30000 * 58.44 / 35.45 = 49,455.6 mg/l NaCl. At
# 47,863.9 ppm the brine weighs (62.368 + 0.438603 * 4.78639 + 1.60074e-3 *
# 4.78639**2) / 62.428 = 1.033254 g/cm3, and 47,863.9 * 1.033254 = 49,455.6.
CHLORIDES_MG_L = 30000.0
NACL_PPM = 47863.9
RMF_75F = 0.0123 + 3647.5 / NACL_PPM ** 0.955  # 0.13606 ohm.m
# The same 30,000 read as NaCl ppm, as the "geolink" profile reads CLFL.
RW_75F_30000_PPM = 0.0123 + 3647.5 / 30000.0 ** 0.955  # 0.20565 ohm.m
TEMPS_F = (150.0, 200.0)


def _arps_f(r1, t1_f, t_f):
    return r1 * (t1_f + 6.77) / (t_f + 6.77)


def _only(*steps):
    """skip_steps that leave only ``steps`` to run."""
    return [step.value for step in WorkflowStep if step.value not in steps]


def _well(temp_unit="degF", n=11):
    well = Well("CHEMISTRY")
    well.depth = np.linspace(2000.0, 2010.0, n)
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(20.0, 120.0, n), "API", "Gamma ray")
    temp = np.linspace(*TEMPS_F, n)
    if temp_unit == "degC":
        temp = (temp - 32.0) / 1.8
    well.add_curve("TEMP", temp, temp_unit, "Formation temperature")
    return well


def _rmf_step(result):
    return next(step for step in result.steps_executed if step.step == "rmf")


def test_chlorides_convert_to_nacl_ppm_with_the_brine_density():
    assert NACL_PER_CHLORIDE == pytest.approx(1.6485, abs=1e-4)
    assert brine_density(0.0) == pytest.approx(62.368 / 62.428)
    assert brine_density(100000.0) == pytest.approx((62.368 + 4.38603 + 0.160074) / 62.428)
    ppm = chlorides_to_nacl_ppm(CHLORIDES_MG_L)
    assert ppm == pytest.approx(NACL_PPM, abs=0.1)
    assert ppm * brine_density(ppm) == pytest.approx(CHLORIDES_MG_L * 58.44 / 35.45)


def test_rmf_from_chlorides_matches_the_hand_calculation():
    config = DynamicWorkflowConfig(rmf_chlorides_mg_l=CHLORIDES_MG_L, skip_steps=_only("rmf"))
    result = DynamicWorkflow(config).run(_well())
    step = _rmf_step(result)

    assert step.method == "mud_filtrate_chlorides_conversion"
    assert step.parameters["mud_filtrate_salinity_ppm"] == pytest.approx(NACL_PPM, abs=0.1)
    assert step.parameters["rmf_at_measured_temp"] == pytest.approx(RMF_75F, rel=1e-5)
    np.testing.assert_allclose(
        result.well.get("RMF").data[[0, -1]],
        [_arps_f(RMF_75F, 75.0, t) for t in TEMPS_F],
        rtol=1e-5,
    )
    assert "x 58.44/35.45" in step.decisions[0] and "brine density" in step.decisions[0]
    with pytest.raises(ValueError, match="rmf_chlorides_mg_l"):
        DynamicWorkflowConfig(rmf_chlorides_mg_l=-1.0)


@pytest.mark.parametrize(
    "profile,method,rmf_75f,setting",
    [
        ("current", "mud_filtrate_chlorides_conversion", RMF_75F, "rmf_chlorides_mg_l"),
        ("geolink", "mud_filtrate_salinity_conversion", RW_75F_30000_PPM, "rmf_salinity_ppm"),
    ],
)
def test_header_clfl_is_read_as_chlorides_except_by_the_benchmark_profile(
    profile, method, rmf_75f, setting
):
    well = _well()
    well.set_parameter("CLFL", CHLORIDES_MG_L)
    well.set_parameter("CLFL__UNIT", "mg/l")

    result = pp.interpret(well, calculation_profile=profile, skip_steps=_only("rmf"))
    step = _rmf_step(result)
    assert step.method == method
    assert getattr(result.config, setting) == CHLORIDES_MG_L
    np.testing.assert_allclose(
        result.well.get("RMF").data[[0, -1]],
        [_arps_f(rmf_75f, 75.0, t) for t in TEMPS_F],
        rtol=1e-5,
    )
    if profile != "current":
        assert step.decisions == [
            "Rmf from CLFL salinity: 30000 ppm -> 0.2057 ohm.m @ 75F and then along the "
            "temperature profile"
        ]

    zoned = run_dynamic_workflow_with_zones(
        well, DynamicWorkflowConfig(calculation_profile=profile, skip_steps=_only("rmf"))
    )
    assert zoned.global_overrides == {setting: CHLORIDES_MG_L}
    assert _rmf_step(zoned.base_result).method == method


@pytest.mark.parametrize(
    "profile,reason",
    [
        ("current", "rmf_salinity_ppm, rmf_chlorides_mg_l or rm, or provide RMF, CLFL or RM in the LAS header"),
        ("geolink", "rmf_salinity_ppm or rm, or provide RMF/RM in the LAS header"),
    ],
)
def test_missing_rmf_source_names_the_chloride_setting_under_current_only(profile, reason):
    well = _well()
    well.remove_curve("TEMP")
    config = DynamicWorkflowConfig(calculation_profile=profile, skip_steps=_only("rmf"))
    skipped = DynamicWorkflow(config).run(well).specific_steps_skipped["rmf"]
    assert skipped == (
        "No mud-filtrate resistivity source: set mud_type='OBM', rmf_at_meas_temp, " + reason
    )


def test_header_chlorides_in_ppm_are_nacl_ppm_without_a_density_step():
    well = _well()
    well.set_parameter("CLFL", 30.0)
    well.set_parameter("CLFL__UNIT", "kppm")

    step = _rmf_step(pp.interpret(well, skip_steps=_only("rmf")))
    assert step.parameters["mud_filtrate_salinity_ppm"] == pytest.approx(30000.0 * 58.44 / 35.45)


@pytest.mark.parametrize(
    "profile,method,temp_key,measured_f",
    [
        ("current", "las_metadata_profile_temperature_correction", "measured_temp_f", 68.0),
        ("geolink", "las_metadata_assumed_reference_temperature", "reference_temp_f", 75.0),
    ],
)
def test_mfst_is_the_rmf_temperature_except_under_the_benchmark_profile(
    profile, method, temp_key, measured_f
):
    well = _well()
    well.set_parameter("RMF", 0.08)
    well.set_parameter("MFST", 20.0)
    well.set_parameter("MFST__UNIT", "degC")

    result = pp.interpret(well, calculation_profile=profile, skip_steps=_only("rmf"))
    step = _rmf_step(result)
    assert step.method == method
    assert step.parameters[temp_key] == pytest.approx(measured_f)
    assert result.well.get("RMF").data[-1] == pytest.approx(_arps_f(0.08, measured_f, 200.0))


@pytest.mark.parametrize("profile,measured_f", [("current", 60.0), ("geolink", 75.0)])
def test_mst_is_the_rm_temperature_except_under_the_benchmark_profile(profile, measured_f):
    well = _well()
    well.set_parameter("RM", 0.1)
    well.set_parameter("MST", 60.0)
    well.set_parameter("MST__UNIT", "degF")

    result = pp.interpret(well, calculation_profile=profile, skip_steps=_only("rmf"))
    step = _rmf_step(result)
    assert step.parameters["rm_measured_temp_f"] == pytest.approx(measured_f)
    # Rmf = 0.75 Rm(T) for fresh mud.
    assert result.well.get("RMF").data[-1] == pytest.approx(0.75 * _arps_f(0.1, measured_f, 200.0))

    zoned = run_dynamic_workflow_with_zones(
        well, DynamicWorkflowConfig(calculation_profile=profile, skip_steps=_only("rmf"))
    )
    if profile == "current":
        assert zoned.global_overrides["rm_meas_temp_f"] == 60.0
    else:
        assert "MST" in zoned.global_unmapped


@pytest.mark.parametrize("order", [("RMFT", "RMT", "MFST", "MST"), ("MFST", "MST", "RMFT", "RMT")])
def test_rmft_and_rmt_win_over_mfst_and_mst_in_any_header_order(order):
    temps_f = {"RMFT": 60.0, "RMT": 50.0, "MFST": 90.0, "MST": 80.0}
    well = _well()
    well.set_parameter("RMF", 0.08)
    for name in order:
        well.set_parameter(name, temps_f[name])
        well.set_parameter(f"{name}__UNIT", "degF")

    interpreted = pp.interpret(well, skip_steps=_only("rmf"))
    zoned = run_dynamic_workflow_with_zones(well, DynamicWorkflowConfig(skip_steps=_only("rmf")))
    assert zoned.global_overrides["rmf_meas_temp_f"] == 60.0
    assert zoned.global_overrides["rm_meas_temp_f"] == 50.0
    for result in (interpreted, zoned.base_result):
        assert _rmf_step(result).parameters["measured_temp_f"] == 60.0


@pytest.mark.parametrize("label", ["F", "degF", "DEGF", "deg F", "\N{DEGREE SIGN}F"])
def test_arps_reads_any_fahrenheit_label(label):
    assert arps_equation(0.5, 75.0, 200.0, label) == pytest.approx(0.5 * 81.77 / 206.77)


def test_arps_uses_the_celsius_constant_for_celsius_and_kelvin():
    t1, t2 = (75.0 - 32.0) / 1.8, (200.0 - 32.0) / 1.8
    expected = 0.5 * (t1 + 21.5) / (t2 + 21.5)
    for label, offset in (("C", 0.0), ("degC", 0.0), ("K", 273.15), ("kelvin", 273.15)):
        assert arps_equation(0.5, t1 + offset, t2 + offset, label) == pytest.approx(expected)


def test_gradient_and_profiles_convert_celsius_labels():
    # 90 degC over 3000 m: 162 degF over 9842.52 ft is 1.64592 degF/100 ft.
    assert calculate_temperature_gradient(20, 0, 110, 3000, "degC", "m") == pytest.approx(1.64592)
    profile = TemperatureProfile.from_two_points(20.0, 0.0, 110.0, 3000.0, "deg C", "m")
    assert profile.surface_temp == pytest.approx(68.0)
    assert profile.reference_temperature == pytest.approx(230.0)
    # The default surface temperature is 68 degF whatever the BHT unit.
    assert estimate_gradient_from_bht(126.0, 3556.0, bht_unit="degC", depth_unit="m") == pytest.approx(
        (258.8 - 68.0) / (3556.0 / 0.3048) * 100.0
    )


def test_salinity_and_sp_helpers_convert_celsius_labels():
    temp_c = (200.0 - 32.0) / 1.8
    rw = rw_from_salinity(30000.0, 200.0)
    assert rw == pytest.approx(_arps_f(RW_75F_30000_PPM, 75.0, 200.0))
    assert rw_from_salinity(30000.0, temp_c, "degC") == pytest.approx(rw)
    assert salinity_from_rw(rw, temp_c, "deg C") == pytest.approx(30000.0)
    # SSP = -K log10(Rmf/Rw) with K = 61 + 0.133 T(degF) = 87.6 at 200 degF.
    assert rw_from_ssp(-50.0, 0.1, temp_c, "degC") == pytest.approx(0.1 / 10 ** (50.0 / 87.6))
    # The default reference temperature is 75 degF (23.89 degC) whatever the unit.
    assert rw_at_reference_temperature(0.05, 200.0) == pytest.approx(0.05 * 206.77 / 81.77)
    assert rw_at_reference_temperature(0.05, temp_c, temp_unit="degC") == pytest.approx(
        0.05 * (temp_c + 21.5) / ((75.0 - 32.0) / 1.8 + 21.5)
    )


def test_temperature_helpers_raise_on_unknown_units():
    calls = (
        lambda: arps_equation(0.5, 75.0, 200.0, "degR"),
        lambda: calculate_temperature_gradient(20.0, 0.0, 110.0, 3000.0, "deg", "m"),
        lambda: TemperatureProfile(gradient=1.0, temp_unit="?"),
        lambda: TemperatureProfile.from_bht(120.0, 3000.0, bht_unit="xyz"),
        lambda: rw_from_salinity(30000.0, 90.0, "Celcius"),
        lambda: UnitConverter.temperature(100.0, "R", "C"),
        lambda: temperature_unit("", strict=True),
    )
    for call in calls:
        with pytest.raises(ValueError, match="temperature unit"):
            call()
    assert temperature_unit("") is None
    assert UnitConverter.temperature(100.0, "degC", "deg F") == pytest.approx(212.0)


def test_rw_and_rmf_modules_convert_a_celsius_temp_curve():
    expected_rw = [_arps_f(RW_75F_30000_PPM, 75.0, t) for t in TEMPS_F]
    for temp_unit in ("degF", "degC"):
        well = _well(temp_unit)
        rw = RwModule(method="salinity").calculate(well, salinity=30000.0).rw
        np.testing.assert_allclose(rw[[0, -1]], expected_rw, rtol=1e-9)
        well.header.rmf = 0.08
        rmf = RmfModule().calculate_from_well(well).rmf
        np.testing.assert_allclose(rmf[[0, -1]], [_arps_f(0.08, 75.0, t) for t in TEMPS_F], rtol=1e-9)


def test_standardizer_converts_celsius_temperature_curves():
    standardized = standardize_curves(_well("degC"))
    temp = standardized.get("TEMP")
    assert temp.unit == "F"
    np.testing.assert_allclose(temp.data, np.linspace(*TEMPS_F, 11))
    assert standardize_curves(_well()).get("TEMP").unit == "degF"
    assert convert_units(100.0, "degC", "F") == pytest.approx(212.0)

    # Celsius curves left unconverted are screened in their own scale (0-204 degC).
    well = _well("degC")
    well.add_curve("TMPC", np.linspace(5.0, 120.0, 11), "degC", "Temperature")
    kept = standardize_curves(well, convert_units=False)
    assert np.isfinite(kept.get("TMPC").data).all()
    assert np.isfinite(kept.get("TEMP").data).all()


@pytest.mark.parametrize(
    "profile,temps_read_f",
    [
        ("current", TEMPS_F),
        ("geolink", ((150.0 - 32.0) / 1.8, (200.0 - 32.0) / 1.8)),
    ],
)
def test_workflow_reads_a_celsius_temp_curve_except_under_the_benchmark_profile(profile, temps_read_f):
    config = DynamicWorkflowConfig(
        rmf_at_meas_temp=0.1, calculation_profile=profile, skip_steps=_only("rmf")
    )
    result = DynamicWorkflow(config).run(_well("degC"))
    np.testing.assert_allclose(
        result.well.get("RMF").data[[0, -1]],
        [_arps_f(0.1, 75.0, t) for t in temps_read_f],
        rtol=1e-9,
    )


def test_context_note_states_the_chloride_conversion():
    step = SimpleNamespace(
        method="mud_filtrate_chlorides_conversion",
        parameters={
            "mud_filtrate_chlorides_mg_l": CHLORIDES_MG_L,
            "mud_filtrate_salinity_ppm": NACL_PPM,
            "brine_density_g_cm3": 1.0333,
            "rmf_at_measured_temp": RMF_75F,
            "measured_temp_f": 75.0,
            "rmf_min": 0.0538,
            "rmf_max": 0.0710,
        },
    )
    note = _rmf_curve_note(step)
    assert "58.44/35.45" in note
    assert "CLFL=30000 mg/l" in note and "NaCl=47864 ppm" in note
