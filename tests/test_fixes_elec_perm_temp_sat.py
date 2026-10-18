"""Regression tests for the electrical / permeability / temperature / saturation fixes."""

import numpy as np
import pytest

from petrophys._internal.benchmark import historical_benchmark
from petrophys.model.well import Well


def _well(curves, unit="m"):
    n = len(next(iter(curves.values())))
    well = Well("t")
    well.depth = np.arange(float(n)) + 1000.0
    well.depth_unit = unit
    for name, values in curves.items():
        well.add_curve(name, np.asarray(values, dtype=float))
    return well


# -- clay volume --------------------------------------------------------------

def test_equal_sp_baselines_give_nan_not_clean_rock():
    from petrophys.interpretation.clay_volume.sp_method import vcl_from_sp

    out = vcl_from_sp(np.array([np.nan, -45.0]), -50.0, -50.0)
    assert np.all(np.isnan(out))
    assert np.isnan(vcl_from_sp(np.nan, -50.0, -50.0))
    with historical_benchmark():
        np.testing.assert_array_equal(vcl_from_sp(np.array([np.nan, -45.0]), -50.0, -50.0), [0.0, 0.0])


def test_flat_sp_channel_does_not_zero_vcl():
    from petrophys.interpretation.clay_volume.vcl_module import VclModule

    gr = [30, 60, 90, 100, 120, 80, 50, 70.0]
    well = _well({"GR": gr, "SP": [-35.0] * 7 + [np.nan]})
    result = VclModule().calculate(well)
    np.testing.assert_allclose(result.vcl, result.vcl_gr)
    assert result.vcl[0] == pytest.approx(0.1)


def test_degenerate_nd_clay_point_gives_nan():
    from petrophys.interpretation.clay_volume.nd_method import vcl_from_nd

    out = vcl_from_nd(np.array([np.nan, 0.30, 0.25]), np.array([2.40, 2.45, np.nan]),
                      den_clay=2.37, neu_clay=0.20)
    assert np.all(np.isnan(out))
    with historical_benchmark():
        np.testing.assert_array_equal(
            vcl_from_nd(np.array([0.3]), np.array([2.4]), den_clay=2.37, neu_clay=0.20), [0.0])


def test_calculate_rcl_accepts_scalars_and_lists():
    from petrophys.interpretation.clay_volume.rcl import calculate_rcl

    assert calculate_rcl(3.3, 0.95) == pytest.approx(3.3)
    assert calculate_rcl([2.0, 3.0], [0.95, 0.95]) == pytest.approx(2.5)
    assert calculate_rcl([2.0, 3.0], [0.5, 0.5]) == 5.0
    with pytest.raises(ValueError):
        calculate_rcl(np.array([2.0]), np.array([0.95]), statistic="mode")


# -- porosity -----------------------------------------------------------------

def test_sonic_porosity_matrix_names():
    from petrophys.interpretation.porosity.sonic_porosity import sonic_porosity

    assert sonic_porosity(70.0, "calcite") == pytest.approx((70 - 47.6) / (189 - 47.6))
    assert sonic_porosity(70.0, "anhydrite") == pytest.approx((70 - 50.0) / (189 - 50.0))
    assert sonic_porosity(70.0, "sandstone") == pytest.approx((70 - 55.5) / (189 - 55.5))
    with pytest.raises(ValueError):
        sonic_porosity(70.0, "granite_typo")


def test_porosity_module_uses_its_matrix_slowness():
    from petrophys.interpretation.porosity.porosity_module import PorosityModule

    well = _well({"RHOB": [2.45, 2.45], "DTC": [70.0, 70.0]})
    phis = PorosityModule(matrix="calcite").calculate(well).phis
    np.testing.assert_allclose(phis, (70 - 47.6) / (189 - 47.6))
    module = PorosityModule()
    module.dtc_matrix = 51.0
    np.testing.assert_allclose(module.calculate(well).phis, (70 - 51.0) / (189 - 51.0))


def test_density_and_neutron_matrix_names():
    from petrophys.interpretation.porosity.density_porosity import calculate_phid
    from petrophys.interpretation.porosity.neutron_porosity import standardize_nphi

    assert calculate_phid(2.45, "anhydrite") == pytest.approx((2.977 - 2.45) / (2.977 - 1.0))
    with pytest.raises(ValueError):
        calculate_phid(2.45, "unknown_rock")
    assert standardize_nphi(0.20, "calcite", "quartz") == pytest.approx(0.24)
    assert standardize_nphi(0.20, "limestone", "limestone") == pytest.approx(0.20)
    with pytest.raises(ValueError):
        standardize_nphi(0.20, "limestone", "granite")


# -- saturation ---------------------------------------------------------------

def test_simandoux_general_n_keeps_missing_inputs_missing():
    from petrophys.interpretation.saturation.simandoux import simandoux

    rt = np.array([np.nan, 15, 15.0])
    phi = np.array([0.2, np.nan, 0.2])
    vcl = np.array([0.2, 0.2, np.nan])
    assert np.all(np.isnan(simandoux(rt, 0.08, phi, vcl, 3.0, 1, 2, 2.4)))
    finite = simandoux(np.array([15.0]), 0.08, np.array([0.2]), np.array([0.2]), 3.0, 1, 2, 2.4)
    assert 0.0 < finite[0] < 1.0
    with historical_benchmark():
        np.testing.assert_array_equal(simandoux(rt, 0.08, phi, vcl, 3.0, 1, 2, 2.4), [1.0, 1.0, 1.0])


def test_infinite_rt_gives_nan_in_shaly_sand_models():
    from petrophys.interpretation.saturation.dual_water import dual_water
    from petrophys.interpretation.saturation.indonesia import indonesia
    from petrophys.interpretation.saturation.nigeria_schlumberger import nigeria_schlumberger
    from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand
    from petrophys.interpretation.saturation.simandoux import simandoux
    from petrophys.interpretation.saturation.waxman_smits import waxman_smits

    assert np.isnan(indonesia(np.inf, 0.05, 0.2, 0.3, 4.0))
    assert np.isnan(simandoux(np.inf, 0.05, 0.2, 0.3, 4.0))
    assert np.isnan(simandoux(np.inf, 0.05, 0.2, 0.3, 4.0, n=2.2))
    assert np.isnan(dual_water(0.25, np.inf, 0.05, 0.2, 0.3, 0.3))
    assert np.isnan(north_sea_shaly_sand(np.inf, 0.05, 0.2, 0.3, 4.0))
    assert np.isnan(nigeria_schlumberger(np.inf, 0.05, 0.2, 0.3, 4.0))
    assert np.isnan(waxman_smits(np.array([0.2]), np.array([np.inf]), 0.1, 0.05, 2.65, 3.83)[0])
    with historical_benchmark():
        assert indonesia(np.inf, 0.05, 0.2, 0.3, 4.0) == 0.0


def test_rtsh_reaches_simandoux_and_indonesia():
    from petrophys.interpretation.saturation.sw_module import SaturationModule, SwModel

    module = SaturationModule()
    args = dict(rt=np.array([5.0]), rw=0.05, rcl=4.0, vcl=np.array([0.4]), phie=np.array([0.15]),
                phit=np.array([0.2]), cec=None, b_coeff=None, rho_ma=2.65, m_star=2.0, rwb=None,
                phi_cl=0.15, vsh_exp=1.6, shale_exp=1.6)
    for model in (SwModel.SIMANDOUX, SwModel.INDONESIA):
        default = module._run_model(model, rtsh=None, **args)
        given = module._run_model(model, rtsh=1.5, **args)
        assert given[0] != pytest.approx(default[0])
        with historical_benchmark():
            np.testing.assert_allclose(module._run_model(model, rtsh=1.5, **args), default)


def test_brooks_corey_holds_given_swir():
    from petrophys.interpretation.saturation.height.saturation_height_module import fit_brooks_corey

    pcs = np.array([2.5, 4, 8, 16, 32, 64.0])
    sw = 0.2 + 0.8 * np.minimum(2 / pcs, 1) ** 1.2
    assert fit_brooks_corey(pcs, sw, swir=0.25).swir == pytest.approx(0.25)
    free = fit_brooks_corey(pcs, sw)
    assert free.swir == pytest.approx(0.2, abs=1e-3)
    assert free.lam == pytest.approx(1.2, abs=1e-2)


def test_pc_from_height_rejects_no_buoyancy():
    from petrophys.interpretation.saturation.height.capillary import pc_from_height

    with pytest.raises(ValueError):
        pc_from_height(10.0, 1.0, 1.1)
    assert pc_from_height(10.0, 1.0, 0.8) == pytest.approx(0.433 * 0.2 * 10)


# -- electrical ---------------------------------------------------------------

def test_salinity_from_rw_invalid_rw_is_nan():
    from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity, salinity_from_rw

    out = salinity_from_rw(np.array([np.nan, 0.1]), 150)
    assert np.isnan(out[0])
    assert rw_from_salinity(out[1], 150) == pytest.approx(0.1)
    assert np.isnan(salinity_from_rw(0.01, 75))
    assert np.isnan(salinity_from_rw(0.0, 75))
    with historical_benchmark():
        assert salinity_from_rw(0.01, 75) > 1e14


def test_rw_from_salinity_nonpositive_is_nan():
    from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity

    assert np.isnan(rw_from_salinity(0, 75))
    out = rw_from_salinity(np.array([-5.0, 50000.0]), 75)
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.131008, rel=1e-4)
    with historical_benchmark():
        assert rw_from_salinity(0, 75) == pytest.approx(3647.5123)


def test_classify_missing_salinity():
    from petrophys.interpretation.electrical.rw.salinity_rw import classify_water_salinity

    assert classify_water_salinity(float("nan")) == "unknown"
    assert classify_water_salinity(np.array([np.nan])) == "unknown"
    assert classify_water_salinity(np.array([np.nan, 50000.0])) == "saline"


def test_rw_from_ssp_scalar_nan_is_nan():
    from petrophys.interpretation.electrical.rw.sp_rw import rw_from_sp, rw_from_ssp

    assert np.isnan(rw_from_ssp(float("nan"), 0.5, 150.0))
    assert np.isnan(rw_from_ssp(-60.0, 0.5, float("nan")))
    assert np.isnan(rw_from_sp(float("nan"), -10.0, 0.5, 150.0))
    assert rw_from_ssp(-60.0, 0.5, 150.0) == pytest.approx(0.0907343, rel=1e-6)
    with historical_benchmark():
        assert rw_from_ssp(float("nan"), 0.5, 150.0) == 0.01


def test_rw_from_rxo_rt_nonpositive_rxo_is_nan():
    from petrophys.interpretation.electrical.rw.sp_rw import rw_from_rxo_rt

    assert np.isnan(rw_from_rxo_rt(0.0, 1.0, 0.5))
    out = rw_from_rxo_rt(np.array([-1.0, 2.0]), np.array([1.0, 1.0]), 0.5)
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.25)


def test_calculate_rw_salinity_with_array_or_zero_temperature():
    from petrophys.interpretation.electrical.rw.rw_module import calculate_rw
    from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity

    out = calculate_rw(method="salinity", salinity=50000, temperature=np.array([150.0, 160.0]))
    np.testing.assert_allclose(out, rw_from_salinity(50000, np.array([150.0, 160.0])))
    assert calculate_rw(method="salinity", salinity=50000, temperature=0.0) == pytest.approx(
        rw_from_salinity(50000, 0.0))


def test_mud_regime_docstring_example():
    from petrophys.interpretation.electrical.invasion.profile import mud_regime

    assert (mud_regime(0.3, 0.05), mud_regime(0.045, 0.05), mud_regime(0.055, 0.05)) == (
        "fresh", "similar", "similar")


# -- permeability -------------------------------------------------------------

def test_one_spike_does_not_rescale_fraction_swirr():
    from petrophys.interpretation.permeability import perm_module

    np.testing.assert_allclose(
        perm_module._as_fraction_if_percent(np.array([0.2, 0.35, 0.25, 2.0])), [0.2, 0.35, 0.25, 2.0])
    np.testing.assert_allclose(
        perm_module._as_fraction_if_percent(np.array([20.0, 35.0, 25.0])), [0.2, 0.35, 0.25])
    with historical_benchmark():
        np.testing.assert_allclose(
            perm_module._as_fraction_if_percent(np.array([0.2, 2.0])), [0.002, 0.02])


def test_coates_denoo_swi_above_one_is_nan():
    from petrophys.interpretation.permeability.coates_denoo import coates_denoo

    assert np.isnan(coates_denoo(0.25, 1.5))
    assert coates_denoo(0.25, 1.0) == pytest.approx(0.0)
    assert coates_denoo(0.25, 0.25) == pytest.approx(10000 * 0.25**4 * 9)
    with historical_benchmark():
        assert coates_denoo(0.25, 1.5) == pytest.approx(4.340277, rel=1e-5)


# -- temperature --------------------------------------------------------------

def test_depth_unit_labels_for_metres():
    from petrophys.interpretation.temperature.temp_gradient import (
        calculate_temperature_gradient,
        estimate_gradient_from_bht,
    )
    from petrophys.interpretation.temperature.temp_profile import TemperatureProfile

    expected = 1.64592
    assert calculate_temperature_gradient(20, 0, 80, 2000, "C", "meters") == pytest.approx(expected)
    assert estimate_gradient_from_bht(80, 2000, bht_unit="C", depth_unit="metres") == pytest.approx(expected)
    assert TemperatureProfile.from_bht(bht=80, td=2000, bht_unit="C", depth_unit="metres").gradient == \
        pytest.approx(expected)
    assert TemperatureProfile.from_two_points(30, 500, 90, 2500, "C", "metres").gradient == \
        pytest.approx(expected)
    profile = TemperatureProfile(gradient=2.0, surface_temp=68)
    assert profile.get_temperature(1000, "meters") == pytest.approx(68 + 2.0 * 1000 / 0.3048 / 100)
    np.testing.assert_allclose(profile.calculate(np.array([1000.0]), "metres").temperature,
                               [68 + 2.0 * 1000 / 0.3048 / 100])
    with pytest.raises(ValueError):
        calculate_temperature_gradient(20, 0, 80, 2000, "C", "furlongs")
    with historical_benchmark():
        assert calculate_temperature_gradient(20, 0, 80, 2000, "C", "meters") == pytest.approx(5.4)


def test_zero_depth_interval_has_no_gradient():
    from petrophys.interpretation.temperature.temp_gradient import calculate_temperature_gradient

    with pytest.raises(ValueError):
        calculate_temperature_gradient(20, 1000, 80, 1000)
    with historical_benchmark():
        assert calculate_temperature_gradient(20, 1000, 80, 1000) == 0.0


def test_unknown_resistivity_temperature_method_raises():
    from petrophys.interpretation.temperature.temp_corrections import correct_resistivity_temperature

    with pytest.raises(ValueError):
        correct_resistivity_temperature(0.85, 75, 150, method="arsp")
    assert correct_resistivity_temperature(0.85, 75, 150) == pytest.approx(0.443353, rel=1e-5)
    assert correct_resistivity_temperature(0.85, 75, 150, method="arp2") == pytest.approx(0.443949, rel=1e-5)
