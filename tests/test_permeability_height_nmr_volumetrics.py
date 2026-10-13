"""Textbook checks for permeability transforms, saturation-height, NMR partitions,
thickness integration and volumetrics."""

from types import SimpleNamespace

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys._internal.analysis_helpers import (
    cell_half_lengths,
    cell_thickness,
    depth_step_array,
)


# ---------------------------------------------------------------------------
# Wyllie-Rose family k = P * phi^Q / Swi^R
# ---------------------------------------------------------------------------
def test_wyllie_rose_presets_match_published_forms():
    from petrophys.interpretation.permeability import WYLLIE_ROSE_PRESETS, timur, wyllie_rose

    # Morris and Biggs (1967) oil: sqrt(k) = 250 phi^3 / Swi. Crain's worked
    # example (phi 0.30, Swirr 0.25) gives about 730 mD.
    assert wyllie_rose(0.30, 0.25) == pytest.approx((250 * 0.30**3 / 0.25) ** 2)
    assert wyllie_rose(0.30, 0.25) == pytest.approx(729.0)
    # Dry gas: sqrt(k) = 79 phi^3 / Swi.
    assert wyllie_rose(0.30, 0.25, "morris_biggs_gas") == pytest.approx((79 * 0.30**3 / 0.25) ** 2)
    # Tixier (1949) carries the same constant as Morris-Biggs oil.
    assert wyllie_rose(0.22, 0.18, "tixier") == pytest.approx(wyllie_rose(0.22, 0.18))
    # Timur (1968) in fractions matches the percent form 0.136 phi%^4.4 / Swi%^2.
    assert wyllie_rose(0.20, 0.25, "timur") == pytest.approx(float(timur(0.20, 0.25)), rel=1e-5)
    assert set(WYLLIE_ROSE_PRESETS) == {"morris_biggs_oil", "morris_biggs_gas", "tixier", "timur"}

    # Hand values: 62500 phi^6 / Swi^2.
    np.testing.assert_allclose(
        wyllie_rose(np.array([0.30, 0.25, 0.10]), np.array([0.10, 0.20, 0.40])),
        [4556.25, 381.4697265625, 0.390625],
    )
    assert wyllie_rose(0.2, 0.5, p=100.0, q=2.0, r=1.0) == pytest.approx(100 * 0.04 / 0.5)
    with pytest.raises(ValueError, match="preset"):
        wyllie_rose(0.2, 0.2, "kozeny")


def test_wyllie_rose_method_uses_the_morris_biggs_oil_constant():
    from petrophys.interpretation.permeability import PermeabilityModule

    well = Well("WR")
    well.depth = np.arange(1000.0, 1005.0)
    well.add_curve("PHIE", np.array([0.30, 0.25, 0.20, 0.15, 0.10]), "v/v")
    result = PermeabilityModule(method="wyllie_rose").calculate(well, swi=0.25)

    phi = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    np.testing.assert_allclose(result.perm, 62500.0 * phi**6 / 0.25**2)
    assert result.metadata["method_metadata"]["equation"] == "62500 * PHIE^6 / SWIRR^2"
    assert result.metadata["method_metadata"]["preset"] == "morris_biggs_oil"


def test_workflow_traces_the_wyllie_rose_equation():
    result = pp.interpret(
        pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230, perm_method="wyllie_rose"
    )
    step = next(s for s in result.steps_executed if s.step == "permeability")
    assert step.parameters["empirical_method"] == "wyllie_rose"
    assert any("k = 62500 * PHIE^6 / SWIRR^2" in text for text in step.decisions)
    phie = np.asarray(result.well.get("PHIE").data, dtype=float)
    perm = np.asarray(result.well.get("PERM").data, dtype=float)
    swirr = step.parameters["swirr"]
    ok = np.isfinite(phie) & (phie > 0.001)
    expected = np.minimum(62500.0 * phie[ok] ** 6 / swirr**2, 10000.0)
    np.testing.assert_allclose(perm[ok], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# lith_vcl: clean rock (VCL = 0) gets a permeability
# ---------------------------------------------------------------------------
def test_lith_vcl_permeability_includes_clean_rock():
    from petrophys.interpretation.permeability import lith_vcl_permeability

    # Sandstone slope 28, intercept -2: 10 ** (28 * PHIT * (1 - 1.2 VCL) - 2).
    assert lith_vcl_permeability(0.15, 0.0) == pytest.approx(10 ** 2.2)
    assert lith_vcl_permeability(0.15, 0.1) == pytest.approx(10 ** (28 * 0.15 * 0.88 - 2))
    assert np.isnan(lith_vcl_permeability(0.15, -0.05))


# ---------------------------------------------------------------------------
# Saturation-height: depth unit and TVD
# ---------------------------------------------------------------------------
def _column(depth, unit):
    well = Well("SH")
    well.depth = depth
    well.depth_unit = unit
    well.add_curve("PERM", np.full(depth.size, 100.0), "mD")
    well.add_curve("PHIE", np.full(depth.size, 0.20), "v/v")
    return well


def _hand_sw_10m_above_fwl():
    # Pc = 0.433 (1.0 - 0.8) h[ft]; J = 0.21645 Pc / (30 cos 30) sqrt(k/phi); Sw = 0.15 J^-0.6
    pc = 0.433 * 0.2 * (10.0 / 0.3048)
    j = 0.21645 * pc / (30.0 * np.cos(np.radians(30.0))) * np.sqrt(100.0 / 0.20)
    return 0.15 * j ** -0.6


def test_saturation_height_converts_metric_height_to_feet():
    from petrophys.interpretation.saturation.height import JFunctionModel, saturation_height_from_logs

    model = JFunctionModel(a=0.15, b=-0.6)
    depth_m = np.arange(2000.0, 2060.5, 0.5)
    i = int(np.flatnonzero(depth_m == 2050.0)[0])
    with pytest.warns(UserWarning, match="tvd_curve"):
        sw_m = saturation_height_from_logs(_column(depth_m, "m"), model, fwl_depth=2060.0)
    with pytest.warns(UserWarning, match="tvd_curve"):
        sw_ft = saturation_height_from_logs(
            _column(depth_m / 0.3048, "ft"), model, fwl_depth=2060.0 / 0.3048
        )
    assert sw_m[i] == pytest.approx(_hand_sw_10m_above_fwl())
    assert sw_m[i] == pytest.approx(0.2197, abs=1e-4)
    np.testing.assert_allclose(sw_m, sw_ft)


def test_saturation_height_uses_a_tvd_curve():
    import warnings

    from petrophys.interpretation.saturation.height import JFunctionModel, saturation_height_from_logs

    model = JFunctionModel(a=0.15, b=-0.6)
    tvd = np.arange(2000.0, 2060.5, 0.5)
    well = _column(tvd * 1.25 + 100.0, "m")      # deviated: MD longer than TVD
    well.add_curve("TVDSS", tvd, "m")
    i = int(np.flatnonzero(tvd == 2050.0)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sw = saturation_height_from_logs(well, model, fwl_depth=2060.0, tvd_curve="TVDSS")
    assert sw[i] == pytest.approx(_hand_sw_10m_above_fwl())
    with pytest.raises(ValueError, match="TVD curve"):
        saturation_height_from_logs(well, model, fwl_depth=2060.0, tvd_curve="TVD")


# ---------------------------------------------------------------------------
# Thickness integration: midpoint cells
# ---------------------------------------------------------------------------
MIXED_DEPTH = np.r_[np.round(np.arange(1000.0, 1010.0001, 0.1), 6), np.arange(1011.0, 1030.0001, 1.0)]


def test_cell_thickness_integrates_mixed_sampling_and_skips_gaps():
    # 0.1 m to 1010 m, then 1.0 m to 1030 m: the cells span 999.95-1030.5 m.
    assert cell_thickness(MIXED_DEPTH).sum() == pytest.approx(30.55)
    assert depth_step_array(MIXED_DEPTH).sum() == pytest.approx(12.1)   # capped steps, kept for diagnostics
    above, below = cell_half_lengths(MIXED_DEPTH)
    i = int(np.flatnonzero(MIXED_DEPTH == 1010.0)[0])
    assert (above[i], below[i]) == pytest.approx((0.05, 0.5))

    # A 49 m hole in a 0.5 m log is a gap: every sample keeps 0.5 m.
    np.testing.assert_allclose(cell_thickness(np.array([1000.0, 1000.5, 1001.0, 1050.0, 1050.5])), 0.5)
    # An isolated sample between two gaps gets the regular step.
    isolated = np.r_[np.arange(0.0, 1.5, 0.15), 30.0, np.arange(55.0, 56.5, 0.15)]
    np.testing.assert_allclose(cell_thickness(isolated), 0.15)
    # Regular sampling, either direction, sums to N steps.
    np.testing.assert_allclose(cell_thickness(np.arange(5.0, 0.0, -0.5)), 0.5)


def test_pay_thickness_follows_mixed_sampling():
    from petrophys.evaluation.intervals import flag_intervals
    from petrophys.evaluation.net_pay import NetPayModule, summarize_pay
    from petrophys.evaluation.summaries import calculate_net_gross, calculate_zone_summary

    n = MIXED_DEPTH.size
    coarse = MIXED_DEPTH > 1010.0
    vcl, phie = np.full(n, 0.1), np.full(n, 0.2)
    sw = np.where(coarse, 0.3, 0.9)
    pay = summarize_pay(vcl=vcl, phie=phie, sw=sw, depth=MIXED_DEPTH,
                        vcl_cutoff=0.4, phie_cutoff=0.08, sw_cutoff=0.6)
    assert pay.gross_thickness == pytest.approx(30.55)
    assert pay.net_pay_thickness == pytest.approx(20.0)      # 1011-1030 m: 1010.5-1030.5
    assert pay.hcpt == pytest.approx(20.0 * 0.2 * 0.7)

    well = Well("MIXED")
    well.depth = MIXED_DEPTH
    well.depth_unit = "m"
    for name, values in (("VCL", vcl), ("PHIE", phie), ("SW", sw)):
        well.add_curve(name, values, "v/v")
    result = NetPayModule(vcl_cutoff=0.4, phie_cutoff=0.08, sw_cutoff=0.6).calculate(well)
    assert result.gross_thickness == pytest.approx(30.55)
    assert result.net_pay_thickness == pytest.approx(20.0)
    for flag in ("NET_ROCK", "NET_RES"):
        well.add_curve(flag, np.ones(n))
    well.add_curve("NET_PAY", coarse.astype(float))
    assert calculate_net_gross(well).gross_thickness == pytest.approx(30.55)
    assert calculate_net_gross(well).net_pay_thickness == pytest.approx(20.0)
    assert calculate_zone_summary(well, 1000.0, 1030.0).net_pay == pytest.approx(20.0)

    table = flag_intervals(coarse, MIXED_DEPTH)
    assert table["top"].tolist() == pytest.approx([1010.5])
    assert table["bottom"].tolist() == pytest.approx([1030.5])


# ---------------------------------------------------------------------------
# Volumetrics: units, weighting, no invented inputs
# ---------------------------------------------------------------------------
def _pay_well(depth, unit, phie, sw):
    well = Well("HCPV")
    well.depth = depth
    well.depth_unit = unit
    well.add_curve("NET_PAY", np.ones(depth.size))
    well.add_curve("PHIE", np.broadcast_to(phie, depth.shape).astype(float), "v/v")
    well.add_curve("SW", np.broadcast_to(sw, depth.shape).astype(float), "v/v")
    return well


def test_hcpv_from_well_converts_metres_to_feet():
    from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well

    depth_m = np.arange(1000.0, 1010.5, 0.5)                # 21 cells of 0.5 m = 10.5 m
    metric = calculate_hcpv_from_well(_pay_well(depth_m, "m", 0.20, 0.30), area_acres=40.0)
    feet = calculate_hcpv_from_well(_pay_well(depth_m / 0.3048, "ft", 0.20, 0.30), area_acres=40.0)
    assert metric.net_pay_ft == pytest.approx(10.5 / 0.3048)
    # OOIP = 7758.4 A h phi (1 - Sw) / Bo
    assert metric.ooip_stb == pytest.approx(7758.4 * 40.0 * (10.5 / 0.3048) * 0.20 * 0.70 / 1.2)
    assert feet.net_pay_ft == pytest.approx(metric.net_pay_ft)
    assert feet.ooip_stb == pytest.approx(metric.ooip_stb)


def test_hcpv_from_well_weights_sw_by_pore_volume_and_needs_inputs():
    from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well

    well = _pay_well(np.array([5000.0, 5001.0]), "ft", np.array([0.10, 0.30]), np.array([0.80, 0.20]))
    result = calculate_hcpv_from_well(well, area_acres=1.0)
    assert result.avg_porosity == pytest.approx(0.20)
    assert result.avg_sw == pytest.approx((0.10 * 0.80 + 0.30 * 0.20) / 0.40)   # 0.35, not 0.50
    # h * phi * (1 - Sw) equals the summed hydrocarbon pore thickness 0.1*0.2 + 0.3*0.8.
    assert result.hcpv_acft == pytest.approx(0.26)

    for missing in ("PHIE", "SW", "NET_PAY"):
        incomplete = _pay_well(np.array([5000.0, 5001.0]), "ft", 0.2, 0.3)
        incomplete.remove_curve(missing)
        with pytest.raises(ValueError, match=missing):
            calculate_hcpv_from_well(incomplete, area_acres=1.0)


# ---------------------------------------------------------------------------
# NMR: 33 ms clastic cutoff and partial-bin partition in log T2
# ---------------------------------------------------------------------------
T2 = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])


def test_t2_cutoff_splits_the_straddling_bin_in_log_t2():
    from petrophys.interpretation.nmr import (
        T2_CUTOFF_CLASTIC,
        NMRData,
        bvi_from_bins,
        ffi_from_bins,
        t2_fraction_below_cutoff,
    )

    assert T2_CUTOFF_CLASTIC == 33.0
    # The 32 ms bin spans sqrt(16*32) = 22.6 to sqrt(32*64) = 45.3 ms.
    share = np.log(33.0 / np.sqrt(16.0 * 32.0)) / np.log(2.0)
    assert t2_fraction_below_cutoff(T2, 33.0)[3] == pytest.approx(share)
    assert share == pytest.approx(0.5444, abs=1e-4)

    bins = np.ones((1, 8))
    assert bvi_from_bins(bins, T2)[0] == pytest.approx(3.0 + share)
    assert ffi_from_bins(bins, T2)[0] == pytest.approx(5.0 - share)
    assert bvi_from_bins(bins, T2, cutoff_ms=32.0)[0] == pytest.approx(3.5)
    assert bvi_from_bins(bins, T2, cutoff_ms=32.0, partial_bins=False)[0] == 3.0
    assert bvi_from_bins(bins, T2, cutoff_ms=33.0, partial_bins=False)[0] == 4.0

    nmr = NMRData(depth=np.array([1.0, 2.0]), mphi=np.array([8.0, 8.0]), bins=np.ones((2, 8)),
                  t2_cutoff_curve=np.array([33.0, 92.0]))
    assert nmr.t2_cutoff == 33.0
    carbonate = 5.0 + np.log(92.0 / np.sqrt(64.0 * 128.0)) / np.log(2.0)
    np.testing.assert_allclose(nmr.bvi_from_bins(), [3.0 + share, carbonate])
    np.testing.assert_allclose(nmr.bvi_from_bins() + nmr.ffi_from_bins(), 8.0)


def test_whole_bin_split_reproduces_the_phillips_summaries():
    from petrophys.interpretation.nmr import NMRData

    nmr = NMRData.from_csv(pp.sample("phillips_nmr_bins"), t2_centers=T2, t2_cutoff=32.0)
    whole = nmr.bvi_from_bins(cutoff_ms=32.0, partial_bins=False)
    np.testing.assert_allclose(whole, nmr.mbvi, atol=0.002)
    # Moving the cutoff from 32 to 33 ms shifts BVI by 4.4% of the 32 ms bin, not all of it.
    shift = nmr.bvi_from_bins(cutoff_ms=33.0) - nmr.bvi_from_bins(cutoff_ms=32.0)
    np.testing.assert_allclose(shift, nmr.bins[:, 3] * np.log(33.0 / 32.0) / np.log(2.0))


# ---------------------------------------------------------------------------
# Formulas stated in docstrings
# ---------------------------------------------------------------------------
def test_coates_calibration_closed_form():
    from petrophys.interpretation.nmr import NMRData
    from petrophys.interpretation.nmr.calibration import calibrate_coates

    depth = np.arange(10.0)
    phi = np.linspace(10.0, 25.0, 10)
    ffi, bvi = phi * 0.7, phi * 0.3
    nmr = NMRData(depth=depth, mphi=phi, mffi=ffi, mbvi=bvi)
    k_core = (phi / 12.0) ** 4 * (ffi / bvi) ** 2 * np.linspace(0.8, 1.25, 10)
    # log c = mean(log phi + (b/a) log(FFI/BVI) - (1/a) log k), a = 4, b = 2
    log_c = np.mean(np.log10(phi) + 0.5 * np.log10(ffi / bvi) - 0.25 * np.log10(k_core))
    calibration = calibrate_coates(nmr, depth, k_core)
    assert calibration.params["c"] == pytest.approx(10.0**log_c)


def test_kozeny_carman_and_arps_docstring_values():
    from petrophys.interpretation.core_analysis.grain_size import GrainSizeData
    from petrophys.interpretation.temperature.temp_corrections import arps_equation

    grains = GrainSizeData(depth=np.array([1.0]), d50=np.array([100.0]))
    # d50^2 phi^3 / (180 (1 - phi)^2) in um^2, times 1013.25 mD/um^2
    expected = 100.0**2 * 0.25**3 / (180.0 * 0.75**2) * 1013.25
    assert float(grains.kozeny_carman_perm(0.25)[0]) == pytest.approx(expected, rel=1e-3)
    assert arps_equation(0.85, t_measured=75, t_formation=150) == pytest.approx(0.85 * 81.77 / 156.77)


def test_north_sea_solves_the_parallel_conductivity_form():
    from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand

    sw = north_sea_shaly_sand(rt=10.0, rw=0.05, phi=0.20, vsh=0.30, rtsh=2.0, vsh_exp=1.6)
    # 1/Rt = phi^m Sw^n / (a Rw) + Vsh^1.6 / Rtsh
    assert 0.20**2 * sw**2 / 0.05 + 0.30**1.6 / 2.0 == pytest.approx(0.1)
    assert sw == pytest.approx(0.184259, abs=1e-6)


# ---------------------------------------------------------------------------
# Overburden stress label: an increment over the analysed interval
# ---------------------------------------------------------------------------
def test_overburden_is_labelled_as_an_interval_increment():
    from petrophys.evaluation.diagnostics import NumericalStatisticalAnalysis

    well = Well("SV")
    well.depth = np.arange(2000.0, 2010.0, 0.5)
    well.depth_unit = "m"
    well.add_curve("RHOB", np.full(20, 2.4), "g/cm3")
    result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    assert "increment" in result.derived_curves["S_V"].description
    assert "S_V increment" in result.derived_curves["SH_MIN"].description
    # S_V adds 2400 kg/m3 * 9.81 * 0.5 m per sample from the top of the interval (MPa);
    # PP_HYDROSTATIC is 1000 * 9.81 * depth plus a fixed 100 m of 1025 kg/m3 seawater.
    sv = result.derived_curves["S_V"].data
    pp_hydrostatic = result.derived_curves["PP_HYDROSTATIC"].data
    np.testing.assert_allclose(sv, np.arange(1, 21) * 2400.0 * 9.81 * 0.5 / 1e6)
    assert pp_hydrostatic[0] == pytest.approx(1000.0 * 9.81 * 2000.0 / 1e6 + 1025.0 * 9.81 * 100.0 / 1e6)


def test_benchmark_context_keeps_the_overburden_note():
    from petrophys.reporting.reference_export import (
        _compact_curve_note,
        _reproduces_benchmark_context,
    )

    well = Well("SV")
    old = "S_V: cumulative overburden stress from RHOB*1000*9.81*depth_step."
    assert _compact_curve_note("S_V", "computed", "", None, well, benchmark=True) == old
    current = _compact_curve_note("S_V", "computed", "", None, well)
    assert "increment" in current and "not the overburden from surface" in current
    archived = {
        "PP_HYDROSTATIC": "PP_HYDROSTATIC: hydrostatic pore pressure from water gradient plus seawater head where applicable.",
        "SH_MIN": "SH_MIN: ((nu/(1-nu))*(S_V-PP_HYDROSTATIC))+PP_HYDROSTATIC.",
    }
    for curve, note in archived.items():
        assert _compact_curve_note(curve, "computed", "", None, well, benchmark=True) == note
    assert "fixed 100 m seawater head" in _compact_curve_note("PP_HYDROSTATIC", "computed", "", None, well)
    sh_min = _compact_curve_note("SH_MIN", "computed", "", None, well)
    assert "not an absolute minimum horizontal stress" in sh_min

    def run(profile):
        return SimpleNamespace(base_result=SimpleNamespace(config=SimpleNamespace(calculation_profile=profile)))

    assert _reproduces_benchmark_context(run("geolink"))
    assert not _reproduces_benchmark_context(run("current"))


@pytest.mark.parametrize("profile", ["geolink", "current"])
def test_context_notes_label_the_overburden_by_profile(profile):
    from petrophys.evaluation.diagnostics import NumericalStatisticalAnalysis
    from petrophys.reporting.reference_export import _build_delivery_computation_notes
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

    well = pp.read(pp.sample("16_7-1")).depth_slice(2400.0, 2500.0)
    original = set(well.curves)
    result = run_dynamic_workflow_with_zones(well, DynamicWorkflowConfig(calculation_profile=profile))
    NumericalStatisticalAnalysis().analyze(result.well, add_curves=True)
    notes = _build_delivery_computation_notes(
        result.well, result, original_curves=original,
        derived_logs_override=["S_V", "PP_HYDROSTATIC"],
        status_targets_override=["S_V", "PP_HYDROSTATIC"],
    )
    sv_notes = [note for note in notes if note.startswith("S_V:")]
    pp_notes = [note for note in notes if note.startswith("PP_HYDROSTATIC:")]
    if profile == "current":
        assert len(sv_notes) == 1 and "increment" in sv_notes[0]
        assert len(pp_notes) == 1 and "fixed 100 m seawater head" in pp_notes[0]
    else:
        assert sv_notes == ["S_V: cumulative overburden stress from RHOB*1000*9.81*depth_step."]
        assert pp_notes == [
            "PP_HYDROSTATIC: hydrostatic pore pressure from water gradient plus seawater head where applicable."
        ]
