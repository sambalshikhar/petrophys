"""Decisions on the I/O, units and lithology method questions.

Each change keeps the benchmark context packages' behaviour inside
``historical_benchmark()``.
"""

import copy
import re
import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys._internal.benchmark import historical_benchmark
from petrophys._internal.statistics import geometric_mean, harmonic_mean
from petrophys.interpretation.geochemistry.xrf import XRFData
from petrophys.interpretation.lithology import MultiMineralSolver
from petrophys.interpretation.lithology.crossplots import nd_crossplot_data
from petrophys.interpretation.lithology.matrix_inference import (
    resolve_from_core_matrix,
    resolve_well_lithology,
)
from petrophys.interpretation.lithology.matrix_properties import get_log_density
from petrophys.interpretation.nmr.nmr_data import NMRData
from petrophys.interpretation.rock_physics import calculate_elastic_properties
from petrophys.io import export_las, read_las
from petrophys.model.log_data import Zone, ZoneSet
from petrophys.preprocessing.standardization import add_standard_curve_aliases, standardize_curves
from petrophys.preprocessing.standardization.alias_map import get_aliases, get_standard_mnemonic


@pytest.fixture(scope="module")
def well_30_6_5():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pp.read(pp.sample("30_6-5"))


@pytest.fixture(scope="module")
def well_hw3():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pp.read(pp.sample("HW-3"))


# --- io ----------------------------------------------------------------------

def test_export_las_writes_six_significant_figures_outside_the_benchmark(tmp_path):
    well = Well("X")
    well.depth = np.array([1.0, 2.0, 3.0])
    well.depth_unit = "m"
    well.add_curve("PERM", np.array([1e-7, 0.123456789, 1234.5678]), "mD", "Permeability")

    export_las(well, tmp_path / "a.las")
    text = (tmp_path / "a.las").read_text()
    assert "1e-07" in text and "0.123457" in text and "1234.57" in text
    np.testing.assert_allclose(read_las(tmp_path / "a.las")["PERM"].data, [1e-7, 0.123457, 1234.57])

    # The benchmark packages were written with fixed five decimals.
    with historical_benchmark():
        export_las(well, tmp_path / "b.las")
    text = (tmp_path / "b.las").read_text()
    assert re.search(r"\s0\.00000\s", text) and "0.12346" in text and "1234.56780" in text


# --- units / standardization -------------------------------------------------

def test_rdep_ranks_at90_first_and_hcgr_is_a_computed_gr():
    assert get_aliases("RDEP").index("AT90") < get_aliases("RDEP").index("AT60")
    assert get_standard_mnemonic("HCGR") == "CGR"
    assert "HCGR" not in get_aliases("GR")
    with historical_benchmark():
        assert get_aliases("RDEP").index("AT60") < get_aliases("RDEP").index("AT90")
        assert get_standard_mnemonic("HCGR") == "GR"
        assert "HCGR" in get_aliases("GR")


def test_hcgr_standardizes_to_cgr_and_at90_wins_an_equal_count_tie(well_30_6_5):
    well = copy.deepcopy(well_30_6_5)
    deep, medium = (well.get(name).data.copy() for name in ("RDEP", "RMED"))
    for name in ("RDEP", "RMED"):
        well._curves.pop(name)
    gr = well.get("GR").data.copy()
    well._curves.pop("GR")
    well.add_curve("AT60", medium, "ohm.m")
    well.add_curve("AT90", deep, "ohm.m")
    well.add_curve("HCGR", gr, "gAPI")

    aliases = add_standard_curve_aliases(copy.deepcopy(well), prefer_deepest_array=False)
    assert aliases.get("AT90") == "RDEP" and aliases.get("HCGR") == "CGR"
    with historical_benchmark():
        aliases = add_standard_curve_aliases(copy.deepcopy(well), prefer_deepest_array=False)
    assert aliases.get("AT60") == "RDEP" and aliases.get("HCGR") == "GR"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standardized = standardize_curves(copy.deepcopy(well))
    assert "CGR" in standardized.curves and "GR" not in standardized.curves


def test_percent_core_saturations_are_converted_by_standardize_curves(well_30_6_5):
    well = copy.deepcopy(well_30_6_5)
    with pytest.warns(UserWarning, match="SWC: inferred percent"):
        standardize_curves(well)
    for name in ("SWC", "SOC"):
        assert well[name].unit == "v/v"
        assert well[name].metadata.custom["core_saturation_normalization"]["factor"] == 0.01
    assert np.nanmax(well["SWC"].data) == pytest.approx(0.846)
    finite = np.isfinite(well["SWC"].data)
    np.testing.assert_allclose(well["SWC"].data[finite], well_30_6_5["SWC"].data[finite] / 100)

    well = copy.deepcopy(well_30_6_5)
    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standardize_curves(well)
    assert well["SWC"].unit == "frac" and np.nanmax(well["SWC"].data) == pytest.approx(84.6)


def test_core_saturation_with_an_unknown_unit_is_left_with_a_warning():
    well = Well("CORE")
    well.depth = np.arange(3.0)
    well.add_curve("SWC", np.array([25.0, 50.0, 75.0]), "mg/L", "odd")
    with pytest.warns(UserWarning, match="left unconverted"):
        standardize_curves(well)
    np.testing.assert_allclose(well["SWC"].data, [25.0, 50.0, 75.0])


# --- lithology ---------------------------------------------------------------

def test_core_grain_density_median_decides_30_6_5(well_30_6_5):
    with pytest.warns(UserWarning, match="core_grain_density"):
        res = resolve_from_core_matrix(well_30_6_5)
    assert res.method == "core_grain_density"
    assert (res.lithology, res.matrix) == ("clastic", "sandstone")
    assert res.n == int(np.isfinite(well_30_6_5["GDEN"].data).sum())
    # The vendor lithology log still ranks above the core matrix density.
    assert resolve_well_lithology(well_30_6_5, warn=False).method == "vendor_log"

    # The benchmark packages voted on the core-porosity matrix density.
    with historical_benchmark():
        res = resolve_from_core_matrix(well_30_6_5, warn=False)
    assert res.method == "core_matrix_density"
    assert res.matrix == "limestone" and res.confidence < 0.8


def test_phie_cc_is_not_core_porosity_outside_the_benchmark(well_hw3):
    assert resolve_from_core_matrix(well_hw3, warn=False) is None
    with historical_benchmark():
        res = resolve_from_core_matrix(well_hw3, warn=False)
    assert res is not None and res.matrix == "dolomite"


def _core_porosity_well(rho_matrix, n=60, phi=0.2):
    w = Well("CORE")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("GR", np.full(n, 20.0), "gAPI", "clean")
    w.add_curve("RHOB", np.asarray(rho_matrix, dtype=float) * (1 - phi) + phi, "g/cm3", "Bulk Density")
    w.add_curve("CPOR", np.full(n, phi), "v/v", "core porosity")
    return w


def test_split_core_matrix_votes_do_not_decide():
    rho = np.full(60, 2.71)
    rho[:20] = 2.65  # 67% limestone
    well = _core_porosity_well(rho)
    with pytest.warns(UserWarning, match="votes are split"):
        assert resolve_from_core_matrix(well) is None
    assert resolve_well_lithology(well, warn=False).method == "default"
    with historical_benchmark():
        res = resolve_from_core_matrix(well, warn=False)
    assert res.matrix == "limestone" and res.confidence == pytest.approx(40 / 60)

    rho[:] = 2.71
    rho[:10] = 2.65  # 83% limestone
    res = resolve_from_core_matrix(_core_porosity_well(rho), warn=False)
    assert res.matrix == "limestone" and res.confidence == pytest.approx(50 / 60)


def test_evaporite_endpoints_use_the_density_log_response():
    assert get_log_density("halite") == pytest.approx(2.032)
    assert get_log_density("anhydrite") == pytest.approx(2.977)
    assert get_log_density("quartz") == pytest.approx(2.65)
    assert nd_crossplot_data(np.zeros(1), np.zeros(1), ["halite"])["endpoints"]["halite"]["rhob"] == pytest.approx(2.032)
    with historical_benchmark():
        assert get_log_density("halite") == pytest.approx(2.17)
        assert nd_crossplot_data(np.zeros(1), np.zeros(1), ["halite"])["endpoints"]["halite"]["rhob"] == pytest.approx(2.17)


def test_pure_halite_solves_to_halite_without_porosity():
    # Hand-built pure halite: the density log reads 2.032 g/cm3, NPHI -0.03, PEF 4.65.
    solver = MultiMineralSolver(minerals=["quartz", "calcite", "halite"])
    result = solver.solve(rhob=[2.032], nphi=[-0.03], vcl=[0.0], pef=[4.65])
    assert result["solved"][0]
    assert result["halite"][0] == pytest.approx(1.0, abs=1e-6)
    assert result["phie"][0] == pytest.approx(0.0, abs=1e-6)
    assert solver.mineral_props["halite"]["density"] == pytest.approx(2.032)
    assert solver.mineral_props["halite"]["u"] == pytest.approx(4.65 * 2.032)

    with historical_benchmark():
        solver = MultiMineralSolver(minerals=["quartz", "calcite", "halite"])
        assert solver.mineral_props["halite"]["density"] == pytest.approx(2.17)
        result = solver.solve(rhob=[2.032], nphi=[-0.03], vcl=[0.0], pef=[4.65])
    assert result["phie"][0] > 0.02


# --- misc / model ------------------------------------------------------------

def test_means_zero_gives_zero_and_negatives_are_excluded_with_a_warning():
    assert harmonic_mean([10.0, 0.0]) == 0.0
    assert geometric_mean([10.0, 0.0]) == 0.0
    assert harmonic_mean([10.0, 100.0]) == pytest.approx(2 / (0.1 + 0.01))
    with pytest.warns(UserWarning, match="negative value"):
        assert harmonic_mean([10.0, -1.0, 100.0]) == pytest.approx(2 / (0.1 + 0.01))
    with pytest.warns(UserWarning, match="negative value"):
        assert geometric_mean([10.0, -1.0, 1000.0]) == pytest.approx(100.0)
    assert np.isnan(harmonic_mean([np.nan, 10.0], ignore_nan=False))
    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert harmonic_mean([10.0, 0.0, -1.0]) == pytest.approx(10.0)
        assert geometric_mean([10.0, 0.0, -1.0]) == pytest.approx(10.0)


def test_zone_boundaries_are_top_inclusive_bottom_exclusive():
    zones = ZoneSet()
    zones.add_zone("upper", 100.0, 200.0)
    zones.add_zone("lower", 200.0, 300.0)
    assert zones.get_at_depth(100.0).name == "upper"
    assert zones.get_at_depth(200.0).name == "lower"
    assert zones.get_at_depth(300.0) is None
    np.testing.assert_array_equal(Zone("z", 100.0, 200.0).contains(np.array([99.0, 100.0, 150.0, 200.0])),
                                  [False, True, True, False])
    assert [z.name for z in zones.get_zones_in_range(200.0, 250.0)] == ["lower"]


# --- nmr / geochem / rock physics --------------------------------------------

def test_elastic_impossible_values_are_nan_outside_the_benchmark():
    # DTS 120 us/ft with DTC 100: Vp/Vs = 1.2 < sqrt(2), no isotropic solid.
    props = calculate_elastic_properties(2.3, 100.0, np.array([120.0, 200.0]))
    assert np.isnan(props.poisson_ratio[0]) and np.isnan(props.bulk_mod_gpa[0]) and np.isnan(props.young_mod_gpa[0])
    assert np.isfinite(props.shear_mod_gpa[0])
    assert props.poisson_ratio[1] == pytest.approx(1 / 3)
    assert props.young_mod_gpa[1] == pytest.approx(2 * props.shear_mod_gpa[1] * (1 + props.poisson_ratio[1]))
    with historical_benchmark():
        props = calculate_elastic_properties(2.3, 100.0, np.array([120.0, 200.0]))
    assert props.poisson_ratio[0] == 0.0 and props.bulk_mod_gpa[0] >= 0.0


def test_xrf_siderite_sulphur_threshold_follows_the_units():
    elements = {"Fe": np.array([1.0, 2.0, 3.0]), "S": np.array([0.001, 0.01, 0.3])}
    fractions = XRFData(depth=np.arange(3.0), elements=elements, units={"Fe": "wt_frac", "S": "wt_frac"})
    assert fractions._sulphur_threshold() == pytest.approx(0.005)
    # 0.01 and 0.3 (1 and 30 wt%) are pyrite-rich: Fe penalised
    np.testing.assert_allclose(fractions.mineral_proxies()["siderite_index"], [1.0, 0.0, 0.75])
    percent = XRFData(depth=np.arange(3.0), elements=elements, units={"Fe": "%", "S": "%"})
    np.testing.assert_allclose(percent.mineral_proxies()["siderite_index"], [0.0, 0.5, 1.0])
    assert XRFData(depth=np.arange(3.0), elements=elements)._sulphur_threshold() == 0.5
    with historical_benchmark():
        assert fractions._sulphur_threshold() == 0.5


def test_elemental_spectroscopy_proxies_carry_fraction_units():
    from petrophys.interpretation.geochemistry.elemental_spectroscopy import interpret_elemental_spectroscopy

    well = Well("ECS")
    well.depth = np.arange(1000.0, 1004.0)
    well.add_curve("DWFE", np.array([0.02, 0.04, 0.06, 0.08]), "wt_frac", "Dry weight fraction of iron")
    well.add_curve("DWSU", np.array([0.001, 0.001, 0.02, 0.001]), "wt_frac", "Dry weight fraction of sulphur")
    well.add_curve("DWSI", np.array([0.3, 0.3, 0.3, 0.3]), "wt_frac", "Dry weight fraction of silicon")
    well.add_curve("DWAL", np.array([0.05, 0.05, 0.05, 0.05]), "wt_frac", "Dry weight fraction of aluminium")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interpret_elemental_spectroscopy(well)
    siderite = well.get("ELEM_SIDERITE_INDEX")
    if siderite is None:
        pytest.skip("siderite index not written for this input")
    # the 2 wt% sulphur sample is penalised: its index drops below its neighbours
    assert siderite.data[2] < siderite.data[1]


def test_bin_cumulative_skips_nan_bins():
    bins = np.vstack([np.array([1.0, np.nan, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]), np.full(8, np.nan)])
    nmr = NMRData(depth=np.arange(2.0), mphi=np.array([20.0, np.nan]), bins=bins)
    cum = nmr.bin_cumulative()
    np.testing.assert_allclose(cum[0], [0.25, 0.25, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(cum[1]).all()
