"""Tests for the electrical, permeability, saturation and clay-volume decisions.

METHOD_DECISIONS.md, "Electrical, permeability, saturation, clay volume". Each
change keeps the benchmark packages' behaviour inside ``historical_benchmark``.
"""

import warnings

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


# -- Rxo/Rt Rw method reads RXO first (#20) ------------------------------------

def test_rxo_rt_rw_reads_the_flushed_zone_rxo_before_rsha():
    from petrophys.interpretation.electrical.rw.rw_module import RwModule

    n = 30
    well = _well({
        "RDEP": np.full(n, 2.0), "RXO": np.full(n, 4.0), "RSHA": np.full(n, 8.0),
        "TEMP": np.full(n, 150.0),
    })
    result = RwModule(method="rxo_rt").calculate(well, rmf=0.5)
    assert result.method == "rxo_rt_clean_water"
    assert result.rw == pytest.approx(0.5 * 2.0 / 4.0)
    with historical_benchmark():
        assert RwModule(method="rxo_rt").calculate(well, rmf=0.5).rw == pytest.approx(0.5 * 2.0 / 8.0)

    # Without RXO the shallow curves still serve.
    well.remove_curve("RXO")
    assert RwModule(method="rxo_rt").calculate(well, rmf=0.5).rw == pytest.approx(0.125)


# -- perm_fallback_method is tried first (#48) ---------------------------------

def test_auto_empirical_order_starts_with_the_fallback_method():
    from petrophys.interpretation.permeability.perm_module import _auto_empirical_methods

    assert _auto_empirical_methods("timur") == ["timur", "coates", "lith_vcl"]
    assert _auto_empirical_methods("lith_vcl") == ["lith_vcl", "coates", "timur"]
    assert _auto_empirical_methods("wyllie_rose") == ["wyllie_rose", "coates", "timur", "lith_vcl"]
    assert _auto_empirical_methods("coates") == ["coates", "timur", "lith_vcl"]
    with historical_benchmark():
        assert _auto_empirical_methods("timur") == ["coates", "timur", "lith_vcl"]
        assert _auto_empirical_methods("wyllie_rose") == ["coates", "wyllie_rose", "timur", "lith_vcl"]


def test_permeability_module_honours_the_fallback_method():
    from petrophys.interpretation.permeability.perm_module import PermeabilityModule
    from petrophys.interpretation.permeability.timur import timur

    well = _well({
        "PHIE": [0.22, 0.18, 0.14, 0.10, 0.16, 0.20],
        "SW": [0.35] * 6,
        "RDEP": [20.0] * 6,
    })
    result = PermeabilityModule(method="auto", fallback_method="timur").calculate(well, swi=0.25)
    assert result.metadata["empirical_method"] == "timur"
    assert result.metadata["empirical_attempted_methods"] == ["timur"]
    np.testing.assert_allclose(result.perm, np.minimum(timur(well.get("PHIE").data, 0.25), 10000.0))

    with historical_benchmark():
        result = PermeabilityModule(method="auto", fallback_method="timur").calculate(well, swi=0.25)
    assert result.metadata["empirical_attempted_methods"] == ["coates"]


# -- reversed SP polarity swaps the picked baselines (#1) ----------------------

def _sp_gr_logs(reversed_sp: bool):
    rng = np.random.default_rng(0)
    n = 400
    gr = np.where((np.arange(n) // 40) % 2 == 0, 30.0, 110.0) + rng.normal(0, 3, n)
    sp = -60.0 + (gr - 30.0) * 0.6 + rng.normal(0, 2, n)  # sand near -60 mV, shale near -12 mV
    return gr, (-sp if reversed_sp else sp)


def test_reversed_sp_swaps_the_auto_picked_baselines_with_a_warning():
    from petrophys.interpretation.clay_volume.vcl_module import VclModule

    gr, sp = _sp_gr_logs(reversed_sp=True)
    well = _well({"GR": gr, "SP": sp})
    with pytest.warns(UserWarning, match="SP polarity is reversed"):
        result = VclModule(use_nd=False).calculate(well)
    sand, shale = gr < 70, gr > 70
    assert np.nanmedian(result.vcl_sp[sand]) < 0.15
    assert np.nanmedian(result.vcl_sp[shale]) > 0.85

    # The benchmark packages took the picks as they are, so VCL_SP inverts.
    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("error")
        old = VclModule(use_nd=False).calculate(well)
    assert np.nanmedian(old.vcl_sp[sand]) > 0.85


def test_normal_sp_keeps_the_auto_picked_baselines():
    from petrophys.interpretation.clay_volume.vcl_module import VclModule

    gr, sp = _sp_gr_logs(reversed_sp=False)
    well = _well({"GR": gr, "SP": sp})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = VclModule(use_nd=False).calculate(well)
    assert np.nanmedian(result.vcl_sp[gr < 70]) < 0.15
    assert np.nanmedian(result.vcl_sp[gr > 70]) > 0.85

    # No GR: nothing to read the polarity from, so no swap.
    well.remove_curve("GR")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        VclModule(use_nd=False).calculate(well)


def test_user_sp_baselines_are_not_swapped():
    from petrophys.interpretation.clay_volume.vcl_module import VclModule

    gr, sp = _sp_gr_logs(reversed_sp=True)
    well = _well({"GR": gr, "SP": sp})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = VclModule(use_nd=False, sp_max=60.0, sp_min=12.0).calculate(well)
    assert np.nanmedian(result.vcl_sp[gr < 70]) < 0.15


# -- Dual Water is the Clavier-Coates-Dumanoir form (#6) -----------------------

def test_dual_water_matches_a_hand_computed_clavier_coates_dumanoir_case():
    from petrophys.interpretation.saturation.dual_water import dual_water

    # PHIT 0.25, Rt 2, Rw 0.05, Rwb 0.02, Vcl 0.30, phi_cl 0.15, a=1, m=n=2.
    # Swb = 0.18; Cw = 20, Cwb = 50; Cw Swt^2 + Swb (Cwb - Cw) Swt - a/(PHIT^m Rt) = 0
    # -> 20 Swt^2 + 5.4 Swt - 8 = 0 -> Swt = 0.51171; Swe = (Swt - Swb)/(1 - Swb) = 0.40452
    swt = (-5.4 + np.sqrt(5.4 ** 2 + 4 * 20 * 8)) / 40
    assert swt == pytest.approx(0.51171, abs=1e-5)
    # Back-substitution: 1/Rt = PHIT^2 Swt^2 [Cw + (Swb/Swt)(Cwb - Cw)]
    assert 0.25 ** 2 * swt ** 2 * (20 + 0.18 / swt * 30) == pytest.approx(0.5, rel=1e-12)
    expected = (swt - 0.18) / (1 - 0.18)
    assert expected == pytest.approx(0.40452, abs=1e-5)
    assert dual_water(0.25, 2.0, 0.05, 0.02, 0.30, 0.15) == pytest.approx(expected, rel=1e-12)
    out = dual_water(np.array([0.25, 0.25]), np.array([2.0, 2.0]), 0.05, 0.02, np.array([0.30, 0.0]), 0.15)
    assert out[0] == pytest.approx(expected, rel=1e-12)
    # Vcl = 0 is Archie on total porosity.
    assert out[1] == pytest.approx(np.sqrt(0.05 / (0.25 ** 2 * 2.0)), rel=1e-12)

    # The wet-zone form of the benchmark packages mixes the waters by Swb, a
    # smaller bound-water share than Swb/Swt, so it reads higher in
    # hydrocarbon-bearing rock.
    with historical_benchmark():
        rwt = 1 / (0.18 / 0.02 + 0.82 / 0.05)
        swt_wet = np.sqrt(rwt / (0.25 ** 2 * 2.0))
        assert dual_water(0.25, 2.0, 0.05, 0.02, 0.30, 0.15) == pytest.approx((swt_wet - 0.18) / 0.82, rel=1e-12)
    assert (swt_wet - 0.18) / 0.82 > expected


def test_dual_water_solves_the_conductivity_equation_for_other_n():
    from petrophys.interpretation.saturation.dual_water import dual_water

    n = 2.3
    swe = float(dual_water(0.25, 2.0, 0.05, 0.02, 0.30, 0.15, n=n))
    swb = 0.18
    swt = swe * (1 - swb) + swb
    assert 0.25 ** 2 * swt ** n * (20 + swb / swt * 30) == pytest.approx(0.5, rel=1e-9)
    assert np.isnan(dual_water(0.25, np.inf, 0.05, 0.02, 0.30, 0.15, n=n))


# -- no VCL curve: PHIE = PHIT and Archie, with warnings (#10, #11) ------------

def test_porosity_without_vcl_writes_phie_equal_to_phit_with_a_warning():
    from petrophys.interpretation.porosity.porosity_module import PorosityModule

    well = _well({"RHOB": [2.45, 2.30, 2.60], "NPHI": [0.15, 0.25, 0.05]})
    with pytest.warns(UserWarning, match="PHIE is written equal to PHIT"):
        result = PorosityModule().calculate(well)
    np.testing.assert_array_equal(result.phie, result.phit)

    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("error")
        old = PorosityModule().calculate(well)
    np.testing.assert_allclose(old.phie, result.phie)

    well.add_curve("VCL", np.array([0.1, 0.5, 0.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with_vcl = PorosityModule().calculate(well)
    assert with_vcl.phie[1] == pytest.approx(with_vcl.phit[1] - 0.5 * 0.15)


def test_saturation_without_vcl_routes_to_archie_with_a_warning():
    from petrophys.interpretation.saturation.archie import archie
    from petrophys.interpretation.saturation.sw_module import SaturationModule

    well = _well({
        "RDEP": [10.0, 20.0, 5.0, 8.0],
        "PHIE": [0.20, 0.25, 0.15, 0.20],
        "PHIT": [0.22, 0.27, 0.17, 0.22],
    })
    expected = archie(well.get("RDEP").data, 0.05, well.get("PHIE").data, 1.0, 2.0, 2.0)
    for model in ("auto", "simandoux", "indonesia"):
        with pytest.warns(UserWarning, match="uses Archie"):
            result = SaturationModule(model=model).calculate(well, rw=0.05, rcl=2.0)
        assert result.model == "archie"
        assert result.model_counts == {"archie": 4}
        np.testing.assert_allclose(result.sw, expected)

    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("error")
        old = SaturationModule(model="simandoux").calculate(well, rw=0.05, rcl=2.0)
    assert old.model == "simandoux"

    well.add_curve("VCL", np.full(4, 0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shaly = SaturationModule(model="simandoux").calculate(well, rw=0.05, rcl=2.0)
    assert shaly.model == "simandoux"


# -- shaly-sand equations have no solution where the shale conducts more (#46) --

def test_north_sea_and_nigeria_return_nan_where_the_shale_term_exceeds_the_rock():
    from petrophys.interpretation.saturation.nigeria_schlumberger import nigeria_schlumberger
    from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand

    # Vsh^1.6 / Rtsh = 0.845 > 1/Rt = 0.2
    assert np.isnan(north_sea_shaly_sand(5.0, 0.05, 0.2, 0.9, 1.0))
    out = north_sea_shaly_sand(np.array([5.0, 5.0]), 0.05, 0.2, np.array([0.9, 0.1]), 1.0)
    assert np.isnan(out[0]) and 0.0 < out[1] < 1.0
    # Vcl^1.6 / sqrt(Rsh) = 0.845 > 1/sqrt(Rt) = 0.447
    assert np.isnan(nigeria_schlumberger(5.0, 0.05, 0.2, 0.9, 1.0))
    out = nigeria_schlumberger(np.array([5.0, 5.0]), 0.05, 0.2, np.array([0.9, 0.1]), 1.0)
    assert np.isnan(out[0]) and 0.0 < out[1] < 1.0
    with historical_benchmark():
        assert north_sea_shaly_sand(5.0, 0.05, 0.2, 0.9, 1.0) == 0.0
        assert nigeria_schlumberger(5.0, 0.05, 0.2, 0.9, 1.0) == 0.0


# -- RmfModule reads RMFT/MFST (#35) -------------------------------------------

def test_rmf_from_well_reads_rmft_then_mfst_for_the_measurement_temperature():
    from petrophys.interpretation.electrical.rmf.rmf_module import RmfModule
    from petrophys.interpretation.temperature.temp_corrections import correct_resistivity_temperature

    well = _well({"TEMP": [150.0, 150.0]})
    well.header.rmf = 0.5
    assert RmfModule().calculate_from_well(well).temperature_measured == 75.0

    well.set_parameter("RMFT", 20.0)
    well.set_parameter("RMFT__UNIT", "degC")
    result = RmfModule().calculate_from_well(well)
    assert result.temperature_measured == pytest.approx(68.0)
    np.testing.assert_allclose(result.rmf, correct_resistivity_temperature(0.5, 68.0, 150.0, "F"))
    with historical_benchmark():
        assert RmfModule().calculate_from_well(well).temperature_measured == 75.0

    well.set_parameter("RMFT", None)
    well.set_parameter("MFST", "80")
    well.set_parameter("MFST__UNIT", "F")
    assert RmfModule().calculate_from_well(well).temperature_measured == pytest.approx(80.0)

    well.header.rm_temp = 90.0
    assert RmfModule().calculate_from_well(well).temperature_measured == 90.0


# -- rw_from_ssp no longer clips (#24) -----------------------------------------

def test_rw_from_ssp_is_not_clipped_outside_the_benchmark():
    from petrophys.interpretation.electrical.rw.sp_rw import rw_from_ssp

    k = 61 + 0.133 * 150.0
    assert rw_from_ssp(60.0, 0.5, 150.0) == pytest.approx(0.5 * 10 ** (60.0 / k))
    assert rw_from_ssp(60.0, 0.5, 150.0) > 2.0
    assert rw_from_ssp(-200.0, 0.5, 150.0) == pytest.approx(0.5 * 10 ** (-200.0 / k))
    assert rw_from_ssp(-200.0, 0.5, 150.0) < 0.01
    out = rw_from_ssp(np.array([60.0, -200.0, np.nan]), 0.5, 150.0)
    assert out[0] > 2.0 and out[1] < 0.01 and np.isnan(out[2])
    assert np.isnan(rw_from_ssp(float("nan"), 0.5, 150.0))
    with historical_benchmark():
        np.testing.assert_allclose(rw_from_ssp(np.array([60.0, -200.0]), 0.5, 150.0), [2.755296, 0.01], rtol=1e-6)
        assert rw_from_ssp(-200.0, 0.5, 150.0) == 0.01
        assert rw_from_ssp(float("nan"), 0.5, 150.0) == 0.01
