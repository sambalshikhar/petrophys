"""Tests for analysis tools promoted from the tutorial notebooks.

Real bundled samples check behaviour on field data (25/5-1 spectral gamma and
core, 30/6-5 SP and tops, COSTA HW-3). Known-answer synthetic arrays check the
arithmetic exactly; they live only in these tests.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.evaluation import (
    SATURATION_MODELS,
    calculate_net_pay,
    compare_saturation_models,
    cutoff_sensitivity,
    parameter_sensitivity,
    summarize_pay,
)
from petrophys.evaluation.intervals import flag_intervals, label_depths
from petrophys.interpretation.core_analysis import (
    CoreData,
    calibrate_porosity,
    core_data_from_well,
    find_core_depth_shift,
    shift_core_depths,
)
from petrophys.interpretation.electrical.rw import sp_rw_in_window
from petrophys.interpretation.geochemistry import (
    TH_U_ENVIRONMENTS,
    classify_th_u_environment,
    interpret_spectral_gamma,
    thorium_clay_index,
    uranium_stripped_gamma_ray,
)
from petrophys.interpretation.saturation import archie
from petrophys.interpretation.temperature import calculate_temperature_profile
from petrophys.model import FormationTops, label_depths as model_label_depths
from petrophys.preprocessing.quality import find_held_readings, held_reading_mask


def _read(name):
    with warnings.catch_warnings():
        # 25/5-1 and 30/6-5 store core saturations in percent labelled 'frac';
        # the reader's note is irrelevant to these tests.
        warnings.simplefilter("ignore", UserWarning)
        return pp.read(pp.sample(name))


def _core(well):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return core_data_from_well(well)


@pytest.fixture(scope="module")
def well_25_5_1():
    well = _read("25_5-1")
    rhob = np.asarray(well.get("RHOB").data, dtype=float)
    well.add_curve("PHID", np.clip((2.65 - rhob) / 1.65, 0.0, 0.5), "v/v")
    return well


@pytest.fixture(scope="module")
def well_30_6_5():
    return _read("30_6-5")


def _synthetic_well(step=0.5):
    well = Well("SYNTH")
    well.depth = np.arange(1000.0, 1100.0, step)
    well.depth_unit = "m"
    well.add_curve("PHID", 0.2 + 0.05 * np.sin(well.depth / 3.0), "v/v")
    return well


# --------------------------------------------------------------------------
# Core depth shift
# --------------------------------------------------------------------------

def test_depth_shift_recovers_a_known_offset_and_applies_it():
    well = _synthetic_well()
    plugs = np.arange(1020.0, 1080.0, 1.0)
    core = CoreData(depth=plugs - 2.0, porosity=0.2 + 0.05 * np.sin(plugs / 3.0))

    scan = find_core_depth_shift(well, core, "PHID", max_shift=5.0)

    assert scan.shift == pytest.approx(2.0)
    assert scan.correlation == pytest.approx(1.0, abs=1e-3)
    assert scan.correlation_unshifted < scan.correlation - 0.1
    assert scan.depth_unit == "m"
    assert len(scan.to_dataframe()) == scan.shifts.size == 21
    assert "shift +2.00 m" in repr(scan)

    tied = shift_core_depths(core, scan)
    np.testing.assert_allclose(tied.depth, plugs)
    assert tied.metadata["depth_shift"] == pytest.approx(2.0)
    np.testing.assert_allclose(core.depth, plugs - 2.0)          # input unchanged
    again = shift_core_depths(tied, -0.5)
    assert again.metadata["depth_shift"] == pytest.approx(1.5)


def test_depth_shift_scans_core_runs_separately():
    well = _synthetic_well()
    upper = np.arange(1010.0, 1040.0, 0.5)
    lower = np.arange(1060.0, 1090.0, 0.5)
    truth = np.r_[upper, lower]
    stored = np.r_[upper - 1.5, lower - 3.0]
    core = CoreData(depth=stored, porosity=0.2 + 0.05 * np.sin(truth / 3.0))

    scan = find_core_depth_shift(well, core, "PHID", max_shift=5.0, run_gap=5.0, min_run_plugs=20)

    assert scan.runs["plugs"].tolist() == [60, 60]
    np.testing.assert_allclose(scan.runs["shift"], [1.5, 3.0])
    per_plug = np.r_[np.full(upper.size, 1.5), np.full(lower.size, 3.0)]
    np.testing.assert_allclose(shift_core_depths(core, per_plug).depth, truth)


def test_depth_shift_ties_the_25_5_1_core(well_25_5_1):
    core = _core(well_25_5_1)
    scan = find_core_depth_shift(well_25_5_1, core, "PHID", run_gap=1.5)

    assert 2.5 < scan.shift < 3.2                         # core stored ~2.9 m shallow
    assert scan.correlation > 0.6 > 0.35 > scan.correlation_unshifted
    assert (scan.runs["shift"] > 1.5).all()               # every main run agrees in sign
    before = calibrate_porosity(well_25_5_1, core, "PHID").r_squared
    after = calibrate_porosity(well_25_5_1, shift_core_depths(core, scan), "PHID").r_squared
    assert after > before + 0.25


def test_depth_shift_handles_permeability_in_log_space(well_25_5_1):
    core = _core(well_25_5_1)
    well = well_25_5_1
    phid = np.asarray(well.get("PHID").data)
    well.add_curve("K_PRED", 10 ** (22.0 * phid - 2.5), "mD")
    scan = find_core_depth_shift(well, core, "K_PRED", core_property="permeability",
                                 log_scale=True)
    assert np.isfinite(scan.correlation) and scan.log_scale


def test_depth_shift_validates_inputs_and_warns_at_the_search_edge():
    well = _synthetic_well()
    plugs = np.arange(1020.0, 1080.0, 1.0)
    core = CoreData(depth=plugs - 4.0, porosity=0.2 + 0.05 * np.sin(plugs / 3.0))
    with pytest.raises(ValueError, match="not found"):
        find_core_depth_shift(well, core, "NOPE")
    with pytest.raises(ValueError, match="no permeability"):
        find_core_depth_shift(well, core, "PHID", core_property="permeability")
    with pytest.raises(ValueError, match="core_property"):
        find_core_depth_shift(well, core, "PHID", core_property="gr")
    with pytest.raises(ValueError, match="step"):
        find_core_depth_shift(well, core, "PHID", step=-1.0)
    with pytest.warns(UserWarning, match="edge"):
        scan = find_core_depth_shift(well, core, "PHID", max_shift=2.0)
    assert scan.at_search_edge
    far = CoreData(depth=np.arange(5000.0, 5020.0), porosity=np.full(20, 0.2))
    with pytest.raises(ValueError, match="covers the cored interval"):
        find_core_depth_shift(well, far, "PHID")
    with pytest.raises(ValueError, match="one per plug"):
        shift_core_depths(core, [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        shift_core_depths(core, np.nan)


# --------------------------------------------------------------------------
# Held readings
# --------------------------------------------------------------------------

def test_held_reading_mask_known_answers():
    values = np.array([61.2, 63.8, 80.19, 80.19, 80.19, 80.19, 0.0, 0.0, np.nan, 5.0])
    np.testing.assert_array_equal(
        held_reading_mask(values, min_run=3),
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        held_reading_mask(values, min_run=2),
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    )
    # NaN breaks a run; a tolerance joins nearly equal readings.
    broken = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    assert not held_reading_mask(broken, min_run=3).any()
    noisy = np.array([5.0, 5.001, 4.999, 5.0, 7.0])
    assert held_reading_mask(noisy, min_run=4, tolerance=0.005).sum() == 3
    assert not held_reading_mask(noisy, min_run=4).any()
    for bad in (1, 2.5):
        with pytest.raises(ValueError, match="min_run"):
            held_reading_mask(values, min_run=bad)
    with pytest.raises(ValueError, match="tolerance"):
        held_reading_mask(values, tolerance=-1.0)


def test_held_readings_at_the_bottom_of_the_25_5_1_spectral_run(well_25_5_1):
    held = find_held_readings(well_25_5_1, ["GR", "THOR", "URAN"])
    summary = held.summary()

    assert summary["held_samples"].to_dict() == {"GR": 57, "THOR": 45, "URAN": 45}
    assert held.shallowest_depth == pytest.approx(3423.43, abs=0.01)
    assert (held.runs["top"] > well_25_5_1.bottom - 10.0).all()   # last metres only
    assert set(held.runs.loc[held.runs["curve"] == "THOR", "value"].round(2)) == {9.6, 0.0}
    assert held.any().sum() == 58          # THOR and URAN add the very last sample
    assert "GR: 57" in repr(held)

    window = find_held_readings(well_25_5_1, ["GR"], bottom=3400.0)
    assert window.shallowest_depth is None and window.summary()["held_samples"].sum() == 0
    with pytest.raises(ValueError, match="not found"):
        find_held_readings(well_25_5_1, ["NOPE"])


# --------------------------------------------------------------------------
# Flags to intervals and depth labels
# --------------------------------------------------------------------------

def test_flag_intervals_known_answers():
    depth = np.arange(2000.0, 2005.0, 0.5)
    pay = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    table = flag_intervals(pay, depth)
    assert table.to_dict("list") == {
        "top": [2000.25, 2002.75, 2003.75],
        "bottom": [2001.75, 2003.25, 2004.75],
        "thickness": [1.5, 0.5, 1.0],
        "samples": [3, 1, 2],
    }
    assert table["thickness"].sum() == pytest.approx(pay.sum() * 0.5)
    assert flag_intervals(pay, depth, max_gap=1.0)["samples"].tolist() == [3, 3]
    assert flag_intervals(pay, depth, min_thickness=0.75)["samples"].tolist() == [3, 2]
    # Descending depth, float flags with NaN, and a Curve-like input.
    float_flag = np.where(pay[::-1] == 1, 1.0, np.nan)
    assert flag_intervals(float_flag, depth[::-1]).equals(table)
    assert flag_intervals(np.zeros(10), depth).empty
    with pytest.raises(ValueError, match="samples"):
        flag_intervals(pay[:-1], depth)


def test_label_depths_tables_rules_and_real_tops(well_30_6_5):
    tops = pp.read_formations(pp.sample("formations_30_6-5"), well=well_30_6_5)
    depths = [2000.0, 2850.0, 2900.0, 2958.0, 100.0]

    with pytest.raises(ValueError, match="overlap"):
        label_depths(depths, tops)
    assert label_depths(depths, tops, overlap="thinnest").tolist() == [
        "Balder Fm.", "Ness Fm.", "Etive Fm.", "Drake Fm.", ""]
    assert label_depths(depths, tops, overlap="thickest").tolist() == [
        "ROGALAND GP.", "BAAT GP.", "BAAT GP.", "DUNLIN GP.", ""]

    formations = pp.read_formations(pp.sample("formations_30_6-5"), well=well_30_6_5,
                                    rank="formation")
    per_sample = label_depths(well_30_6_5, formations)
    assert per_sample.shape == (well_30_6_5.n_samples,)
    brent = (well_30_6_5.depth >= 2865.0) & (well_30_6_5.depth < 2958.0)
    assert set(per_sample[brent]) == {"Etive Fm."}

    lithology = pp.read_lithology_intervals(pp.sample("lithology_30_6-5"), well=well_30_6_5)
    assert label_depths([2900.0, 2300.0], lithology).tolist() == ["sandstone", "chalk"]

    collection = FormationTops.from_dataframe(tops)
    assert model_label_depths([2850.0], collection, overlap="thinnest").tolist() == ["Ness Fm."]
    assert label_depths([2850.0], collection.intervals, label="lithology",
                        overlap="thinnest").tolist() == ["Sandstone"]
    with pytest.raises(ValueError, match="overlap must be"):
        label_depths(depths, tops, overlap="first")
    with pytest.raises(ValueError, match="label"):
        label_depths(depths, pd.DataFrame({"top": [1.0], "bottom": [2.0], "x": ["a"]}))


# --------------------------------------------------------------------------
# Pay summary and cutoff sensitivity
# --------------------------------------------------------------------------

DEPTH = np.arange(2000.0, 2010.0, 1.0)
VCL = np.array([0.1, 0.1, 0.2, 0.6, 0.6, 0.1, 0.1, 0.3, 0.1, 0.1])
PHIE = np.array([0.25, 0.22, 0.18, 0.05, 0.04, 0.20, 0.21, 0.12, 0.06, 0.24])
SW = np.array([0.20, 0.30, 0.45, 1.00, 1.00, 0.35, 0.80, 0.50, 0.90, 0.25])


def test_summarize_pay_known_answer_window_and_defaults():
    pay = summarize_pay(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH, depth_unit="m",
                        vcl_cutoff=0.4, phie_cutoff=0.08, sw_cutoff=0.6)
    assert pay.gross_thickness == 10.0
    assert (pay.net_rock_thickness, pay.net_reservoir_thickness, pay.net_pay_thickness) == (
        8.0, 7.0, 6.0)
    assert pay.ntg_reservoir == pytest.approx(0.7) and pay.ntg_pay == pytest.approx(0.6)
    assert pay.hcpt == pytest.approx(0.823)
    assert pay.avg_phie_pay == pytest.approx(1.21 / 6)
    assert pay.n_pay_samples == 6 and pay.net_pay.sum() == 6
    assert pay.net_reservoir.sum() == 7
    assert "net pay 6.00 m" in repr(pay)
    assert "net_pay" not in pay.to_dict() and "net_reservoir" not in pay.to_dict()

    window = summarize_pay(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH, top=2003.0, bottom=2006.0)
    assert window.gross_thickness == 4.0 and window.net_pay_thickness == 1.0
    standard = summarize_pay(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH)
    assert (standard.vcl_cutoff, standard.phie_cutoff, standard.sw_cutoff) == (0.4, 0.08, 0.6)
    dry = summarize_pay(vcl=VCL, phie=PHIE, sw=np.ones(10), depth=DEPTH)
    assert dry.net_pay_thickness == 0.0 and dry.hcpt == 0.0 and np.isnan(dry.avg_sw_pay)

    with pytest.raises(ValueError, match="samples"):
        summarize_pay(vcl=VCL[:5], phie=PHIE, sw=SW, depth=DEPTH)
    with pytest.raises(ValueError, match="Pass sw="):
        summarize_pay(vcl=VCL, phie=PHIE, depth=DEPTH)
    with pytest.raises(ValueError, match="window"):
        summarize_pay(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH, top=3000.0)


@pytest.fixture(scope="module")
def interpreted_hw3():
    result = pp.interpret(_read("HW-3"), lithology_domain="carbonate")
    result.raise_for_errors()
    return result.well


def test_summarize_pay_matches_calculate_net_pay_on_a_real_well(interpreted_hw3):
    cutoffs = dict(vcl_cutoff=0.30, phie_cutoff=0.10, sw_cutoff=0.60)
    expected = calculate_net_pay(interpreted_hw3, lithology="carbonate", **cutoffs)
    pay = summarize_pay(interpreted_hw3, **cutoffs)
    assert pay.net_pay_thickness == pytest.approx(expected.net_pay_thickness)
    assert pay.net_reservoir_thickness == pytest.approx(expected.net_reservoir_thickness)
    assert pay.gross_thickness == pytest.approx(expected.gross_thickness)
    assert pay.depth_unit == "ft" and pay.hcpt > 0


def test_cutoff_sensitivity_grid(interpreted_hw3):
    grid = cutoff_sensitivity(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH,
                              vcl_cutoffs=[0.2, 0.4], sw_cutoffs=[0.4, 0.9])
    net = grid.pivot(index="vcl_cutoff", columns="sw_cutoff", values="net_pay_thickness")
    assert net.to_dict("index") == {0.2: {0.4: 4.0, 0.9: 5.0}, 0.4: {0.4: 4.0, 0.9: 7.0}}
    assert (grid["phie_cutoff"] == 0.08).all()          # unset cutoff: clastic standard

    sw_cuts = np.round(np.arange(0.4, 0.91, 0.1), 2)
    real = cutoff_sensitivity(interpreted_hw3, vcl_cutoffs=0.3, phie_cutoffs=0.1,
                              sw_cutoffs=sw_cuts, lithology="carbonate")
    assert len(real) == sw_cuts.size and real.attrs["depth_unit"] == "ft"
    assert np.all(np.diff(real["net_pay_thickness"]) >= 0)   # looser Sw cutoff, more pay
    base = real.loc[np.isclose(real["sw_cutoff"], 0.6)].iloc[0]
    single = summarize_pay(interpreted_hw3, vcl_cutoff=0.3, phie_cutoff=0.1, sw_cutoff=0.6)
    assert base["net_pay_thickness"] == pytest.approx(single.net_pay_thickness)
    assert base["hcpt"] == pytest.approx(single.hcpt)
    with pytest.raises(ValueError, match="finite"):
        cutoff_sensitivity(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH, sw_cutoffs=[0.5, np.nan])


# --------------------------------------------------------------------------
# Parameter sensitivity
# --------------------------------------------------------------------------

def test_parameter_sensitivity_scalar_mapping_and_object_outputs():
    def net_pay(rw=0.07, m=2.0):
        return 100.0 - 300.0 * rw - 10.0 * m

    table = parameter_sensitivity(net_pay, {
        "m 1.8 / 2.2": ({"m": 1.8}, {"m": 2.2}),
        "Rw 0.05 / 0.09": ({"rw": 0.05}, {"rw": 0.09}),
    })
    assert table["case"].tolist() == ["Rw 0.05 / 0.09", "m 1.8 / 2.2"]    # widest first
    np.testing.assert_allclose(table["swing"], [12.0, 4.0])
    assert table.attrs["base"] == {"value": 59.0}
    assert table.loc[0, "low_setting"] == "rw=0.05"

    def pay(sw_cutoff=0.6):
        return summarize_pay(vcl=VCL, phie=PHIE, sw=SW, depth=DEPTH, sw_cutoff=sw_cutoff)

    both = parameter_sensitivity(pay, {"Sw cutoff": ({"sw_cutoff": 0.4}, {"sw_cutoff": 0.8})},
                                 outputs=["net_pay_thickness", "hcpt"])
    assert both["output"].tolist() == ["net_pay_thickness", "hcpt"]
    assert both.loc[0, ["low", "base", "high"]].tolist() == [4.0, 6.0, 6.0]

    mapping = parameter_sensitivity(lambda x=1.0: {"a": x, "b": 2 * x, "label": "ignored"},
                                    {"x": ({"x": 0.0}, {"x": 3.0})})
    assert mapping.set_index("output")["swing"].to_dict() == {"a": 3.0, "b": 6.0}

    with pytest.raises(ValueError, match="non-empty"):
        parameter_sensitivity(net_pay, {})
    with pytest.raises(ValueError, match="pair"):
        parameter_sensitivity(net_pay, {"m": {"m": 1.8}})
    with pytest.raises(TypeError, match="number"):
        parameter_sensitivity(lambda **_: "text", {"m": ({}, {})})
    with pytest.raises(ValueError, match="did not return"):
        parameter_sensitivity(pay, {"s": ({}, {})}, outputs=["missing"])


# --------------------------------------------------------------------------
# Saturation-model comparison
# --------------------------------------------------------------------------

def test_saturation_models_reduce_to_archie_in_clean_rock():
    rt = np.array([20.0, 12.0, 5.0, 2.0, 30.0])
    phie = np.array([0.22, 0.20, 0.18, 0.15, 0.24])
    result = compare_saturation_models(rt=rt, phie=phie, vcl=np.zeros(5), depth=np.arange(5.0),
                                       rw=0.07, rsh=3.0, a=0.8, m=2.0, n=2.0)
    expected = np.clip(archie(rt, 0.07, phie, 0.8, 2.0, 2.0), 0.0, 1.0)
    assert list(result.sw) == list(SATURATION_MODELS)
    for model in SATURATION_MODELS:
        np.testing.assert_allclose(result.sw[model], expected, rtol=1e-9)
    assert result.table["hcpt"].nunique() == 1

    with pytest.raises(ValueError, match="Unknown saturation model"):
        compare_saturation_models(rt=rt, phie=phie, vcl=np.zeros(5), depth=np.arange(5.0),
                                  rw=0.07, rsh=3.0, models=["waxman_smits"])
    with pytest.raises(ValueError, match="rsh"):
        compare_saturation_models(rt=rt, phie=phie, vcl=np.zeros(5), depth=np.arange(5.0),
                                  rw=0.07, models=["indonesia"])
    only_archie = compare_saturation_models(rt=rt, phie=phie, vcl=np.zeros(5),
                                            depth=np.arange(5.0), rw=0.07, models=["archie"])
    assert list(only_archie.table.index) == ["archie"]
    with pytest.raises(ValueError, match="rw"):
        compare_saturation_models(rt=rt, phie=phie, vcl=np.zeros(5), depth=np.arange(5.0),
                                  rw=-1.0, models=["archie"])


def test_saturation_models_on_the_30_6_5_brent(well_30_6_5):
    from petrophys.interpretation.clay_volume import vcl_from_gr
    from petrophys.interpretation.porosity import (
        density_porosity, effective_porosity, total_porosity,
    )

    brent = well_30_6_5.depth_slice(2841.0, 2958.0)
    gr = brent.get("GR").data
    vcl = np.clip(vcl_from_gr(gr, np.nanpercentile(gr, 5), np.nanpercentile(gr, 95),
                              method="larionov_older"), 0.0, 1.0)
    phit = np.clip(total_porosity(brent.get("NPHI").data,
                                  density_porosity(brent.get("RHOB").data, 2.65, 1.0),
                                  method="adaptive"), 0.001, 0.45)
    phie = np.clip(effective_porosity(phit, vcl, phi_clay=0.15), 0.001, 0.40)

    result = compare_saturation_models(brent, phie=phie, vcl=vcl, rw=0.07, rsh=3.0, a=0.8)

    table = result.table
    assert table["mean_sw"].idxmax() == "archie"                 # clay-blind model reads wettest
    assert table.loc["archie", "hcpt"] < table.drop(index="archie")["hcpt"].min()
    assert result.depth_unit == "m" and result.depth.size == brent.n_samples
    assert result.parameters["sw_cutoff"] == 0.6
    assert "5 models" in repr(result)


# --------------------------------------------------------------------------
# SP-derived Rw in a window
# --------------------------------------------------------------------------

def test_sp_rw_in_a_local_window_avoids_baseline_drift(well_30_6_5):
    temp_f = calculate_temperature_profile(
        well_30_6_5, bht=258.8, bht_unit="F", surface_temp=68.0, td=well_30_6_5.bottom,
    ).temperature
    kwargs = dict(rmf=0.186, rmf_temp_f=75.0, temperature_f=temp_f, reference_temp_f=224.0)

    whole = sp_rw_in_window(well_30_6_5, **kwargs)
    local = sp_rw_in_window(well_30_6_5, top=2800.0, bottom=2958.0, **kwargs)

    assert whole.ssp_mv < -40.0                    # mostly baseline drift
    assert -8.0 < local.ssp_mv < -3.0              # small deflection, similar salinities
    assert local.rw == pytest.approx(0.058, abs=0.002)
    assert local.rw > 2.5 * whole.rw
    assert local.top == pytest.approx(2800.0, abs=0.2) and local.bottom <= 2958.0
    assert local.window_temp_f == pytest.approx(np.median(temp_f[
        (well_30_6_5.depth >= 2800.0) & (well_30_6_5.depth <= 2958.0)]))

    at_window = sp_rw_in_window(well_30_6_5, rmf=0.186, rmf_temp_f=75.0, temperature_f=temp_f,
                                top=2800.0, bottom=2958.0)
    assert at_window.reference_temp_f == pytest.approx(at_window.window_temp_f)
    assert at_window.rw == pytest.approx(at_window.rw_window)

    with pytest.raises(ValueError, match="samples"):
        sp_rw_in_window(well_30_6_5, rmf=0.186, rmf_temp_f=75.0, temperature_f=temp_f[:10])
    with pytest.raises(ValueError, match="window"):
        sp_rw_in_window(well_30_6_5, top=9000.0, **kwargs)
    with pytest.raises(ValueError, match="rmf"):
        sp_rw_in_window(well_30_6_5, rmf=-1.0, rmf_temp_f=75.0, temperature_f=200.0)
    no_sp = Well("NO-SP")
    no_sp.depth = np.arange(10.0)
    with pytest.raises(ValueError, match="SP"):
        sp_rw_in_window(no_sp, rmf=0.1, rmf_temp_f=75.0, temperature_f=150.0)


# --------------------------------------------------------------------------
# Thorium / uranium indicators
# --------------------------------------------------------------------------

def test_th_u_environment_known_answers():
    thorium = np.array([12.0, 11.0, 2.4, 9.0, 4.0, 14.0, -0.3, np.nan])
    uranium = np.array([1.2, 6.0, 0.2, 2.0, 2.0, 2.0, 1.0, 1.0])
    env = classify_th_u_environment(thorium, uranium)

    assert env.environment.tolist() == [
        "continental oxidising", "marine reducing", "", "marine",
        "marine reducing", "marine", "", ""]            # Th/U = 2 and 7 fall in the lower class
    assert env.rated.sum() == 5
    table = env.summary({"all": np.ones(8, bool), "first two": np.arange(8) < 2})
    assert table.loc["all", "samples"] == 5
    assert table.loc["first two", "median Th/U"] == pytest.approx((10.0 + 11.0 / 6.0) / 2)
    shares = table.loc["all", [f"{label} (%)" for label in TH_U_ENVIRONMENTS]]
    np.testing.assert_allclose(shares, [40.0, 40.0, 20.0])
    assert classify_th_u_environment(thorium, uranium, uranium_floor_ppm=3.0).rated.sum() == 1

    with pytest.raises(ValueError, match="values"):
        classify_th_u_environment(thorium, uranium[:3])
    with pytest.raises(ValueError, match="thresholds"):
        classify_th_u_environment(thorium, uranium, thresholds=(7.0, 2.0))
    with pytest.raises(ValueError, match="values for"):
        env.summary({"bad": np.ones(3, bool)})


def test_thorium_clay_index_and_uranium_stripping_known_answers():
    ith = thorium_clay_index([2.0, 6.0, 10.0, 14.0, 20.0, np.nan], clean_ppm=2.0, shale_ppm=14.0)
    np.testing.assert_allclose(ith.index[:5], [0.0, 1 / 3, 2 / 3, 1.0, 1.0])
    assert np.isnan(ith.index[5])
    picked = thorium_clay_index(np.arange(0.0, 101.0), mask=np.arange(101) >= 50)
    assert (picked.clean_ppm, picked.shale_ppm) == (52.5, 97.5)
    with pytest.raises(ValueError, match="above the clean"):
        thorium_clay_index([1.0, 2.0], clean_ppm=5.0, shale_ppm=5.0)

    np.testing.assert_allclose(
        uranium_stripped_gamma_ray([80.0, 150.0], [2.0, 10.0]), [63.82, 69.1])
    np.testing.assert_allclose(
        uranium_stripped_gamma_ray([80.0], [2.0], api_per_ppm_uranium=5.0), [70.0])
    with pytest.raises(ValueError, match="values"):
        uranium_stripped_gamma_ray([80.0, 1.0], [2.0])
    with pytest.raises(ValueError, match="positive"):
        uranium_stripped_gamma_ray([80.0], [2.0], api_per_ppm_uranium=0.0)


def test_spectral_gamma_indicators_on_25_5_1(well_25_5_1):
    held = find_held_readings(well_25_5_1, ["GR", "THOR", "URAN"])
    depth = well_25_5_1.depth
    gr = np.asarray(well_25_5_1.get("GR").data)
    thor = well_25_5_1.get("THOR")
    uran = well_25_5_1.get("URAN")
    usable = (np.isfinite(gr) & np.isfinite(thor.data) & np.isfinite(uran.data)
              & (depth < held.shallowest_depth))

    sg = interpret_spectral_gamma(well_25_5_1, add_curves=False, alias_overrides={"SGR": ["GR"]})
    u_flag = usable & (sg.derived["SGR_URANIUM_EXCESS_FLAG"] == 1)
    env = classify_th_u_environment(thor, uran)          # Curve inputs are accepted
    table = env.summary({"uranium-dominated": u_flag, "rest": usable & ~u_flag})
    assert table.loc["uranium-dominated", "median Th/U"] < 2.0 < table.loc["rest", "median Th/U"]
    assert table.loc["uranium-dominated", "marine reducing (%)"] > 80.0

    ith = thorium_clay_index(thor, mask=usable)
    assert 1.5 < ith.clean_ppm < 3.0 and 14.0 < ith.shale_ppm < 17.0
    stripped = uranium_stripped_gamma_ray(gr, uran)
    np.testing.assert_allclose(stripped[usable], (gr - sg.derived["SGR_URANIUM_API"])[usable])
    # Stripping uranium lowers the GR shale baseline towards a normal shale value.
    assert np.percentile(stripped[usable], 95) < 0.6 * np.percentile(gr[usable], 95)
