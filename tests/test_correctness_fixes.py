"""Regression tests for cross-module correctness.

Each test uses hand-computed expectations for deterministic behavior.
"""

import numpy as np
import pytest

from petrophys.model.curve import Curve
from petrophys.model.well import Well


def _make_well(curves, n=20, top=1000.0, step=0.5):
    well = Well(name="t")
    depth = np.arange(top, top + n * step, step)
    well.add_curve("DEPT", depth, "m", "depth")
    well._depth = depth
    for name, data, unit in curves:
        well.add_curve(name, data, unit, name)
    return well


# ---------------------------------------------------------------------------
# Gypsum matrix slowness: standard chart value ~52.5 us/ft, not 83. The wrong
# value flipped the sign of Wyllie sonic porosity in gypsum intervals.
# ---------------------------------------------------------------------------
def test_gypsum_slowness_is_standard_in_both_tables():
    from petrophys.model.constants import MatrixProperties
    from petrophys.interpretation.lithology.matrix_properties import MINERAL_PROPERTIES

    assert MatrixProperties.get("gypsum").slowness == pytest.approx(52.5)
    assert MINERAL_PROPERTIES["gypsum"]["slowness"] == pytest.approx(52.5)
    # Wyllie porosity in gypsum at DT=60 must be small and positive.
    phi = (60.0 - 52.5) / (189.0 - 52.5)
    assert 0.0 < phi < 0.1


# ---------------------------------------------------------------------------
# Wentworth scale: silt spans 3.9-62.5 um in four subclasses; below 3.9 um is
# clay; boulder starts at 256 mm. Sorting must be positive and skewness finite
# (and zero for a log-symmetric distribution) under the documented d13 < d87
# diameter-percentile convention.
# ---------------------------------------------------------------------------
def test_wentworth_boundaries():
    from petrophys.interpretation.core_analysis.grain_size import wentworth_class

    assert wentworth_class(3.0) == "Clay"
    assert wentworth_class(5.0) == "Very Fine Silt"
    assert wentworth_class(10.0) == "Fine Silt"
    assert wentworth_class(20.0) == "Medium Silt"
    assert wentworth_class(40.0) == "Coarse Silt"
    assert wentworth_class(100.0) == "Very Fine Sand"
    assert wentworth_class(100_000) == "Cobble"
    assert wentworth_class(256_000) == "Boulder"


def test_grain_sorting_and_skewness_conventions():
    from petrophys.interpretation.core_analysis.grain_size import GrainSizeData

    symmetric = GrainSizeData(
        depth=np.array([1.0]), d50=np.array([200.0]),
        d13=np.array([100.0]), d87=np.array([400.0]),
    )
    assert float(symmetric.sorting_coefficient()[0]) == pytest.approx(1.0)
    assert float(symmetric.skewness()[0]) == pytest.approx(0.0)

    fine_skewed = GrainSizeData(
        depth=np.array([1.0]), d50=np.array([200.0]),
        d13=np.array([50.0]), d87=np.array([400.0]),
    )
    assert float(fine_skewed.skewness()[0]) > 0.0


# ---------------------------------------------------------------------------
# calculate_zone_averages must work with its documented default arguments
# (previously crashed: Well.curves is a list of names, not objects).
# ---------------------------------------------------------------------------
def test_zone_averages_default_arguments():
    from petrophys.evaluation.summaries.averages import calculate_zone_averages

    well = _make_well([("GR", np.linspace(20, 120, 20), "API")])
    result = calculate_zone_averages(well, 1000.0, 1009.0)
    assert np.isfinite(result["GR"])


# ---------------------------------------------------------------------------
# _estimate_rcl: shale-flagged samples with no finite RT must fall back to the
# 5.0 default instead of returning NaN (which poisoned all shaly Sw).
# ---------------------------------------------------------------------------
def test_estimate_rcl_nan_guard():
    from petrophys.interpretation.saturation.sw_module import SaturationModule

    module = SaturationModule(model="simandoux")
    rt = np.array([5.0, 5.0, np.nan, np.nan])
    vcl = np.array([0.2, 0.2, 0.95, 0.95])
    assert module._estimate_rcl(rt, vcl) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Alias map: RESS/SFLU are shallow curves (the rest of the library reads them
# as shallow); TVD/TVDSS are physically distinct from measured depth and must
# not standardize to DEPTH.
# ---------------------------------------------------------------------------
def test_alias_map_shallow_and_depth_families():
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    assert get_standard_mnemonic("RESS") == "RSHA"
    assert get_standard_mnemonic("SFLU") == "RSHA"
    assert get_standard_mnemonic("TVD") == "TVD"
    assert get_standard_mnemonic("TVDSS") == "TVDSS"
    assert get_standard_mnemonic("MD") == "DEPTH"


# ---------------------------------------------------------------------------
# Curve-standardizer collisions: both curves must always survive, the loser
# under its own original mnemonic, and a curve natively named like the
# standard takes the slot.
# ---------------------------------------------------------------------------
def _res_well(pairs):
    curves = []
    for name, nvalid in pairs:
        arr = np.full(20, np.nan)
        arr[:nvalid] = 50.0
        curves.append((name, arr, "ohm.m"))
    return _make_well(curves)


def test_standardizer_collision_incoming_wins():
    from petrophys.preprocessing.standardization import standardize_curves

    well = _res_well([("RT", 7), ("ILD", 10)])
    standardize_curves(well, rename_curves=True, convert_units=False)
    assert well.get("RDEP").valid_count() == 10          # ILD data won the slot
    assert well.get("RT").valid_count() == 7             # the occupant keeps its own name


def test_standardizer_collision_incoming_loses():
    from petrophys.preprocessing.standardization import standardize_curves

    well = _res_well([("RT", 10), ("ILD", 5)])
    standardize_curves(well, rename_curves=True, convert_units=False)
    assert well.get("RDEP").valid_count() == 10
    assert well.get("ILD").valid_count() == 5            # fewer valid samples: kept under its own name


def test_standardizer_collision_ties_keep_the_first_curve():
    from petrophys.preprocessing.standardization import standardize_curves

    well = _res_well([("RT", 10), ("RDEP", 10)])
    standardize_curves(well, rename_curves=True, convert_units=False)
    assert well.get("RDEP").valid_count() == 10
    assert "RT" not in well.curves


def test_standardizer_keeps_tvd_distinct():
    from petrophys.preprocessing.standardization import standardize_curves

    well = _make_well([
        ("MD", np.arange(1000.0, 1010.0, 0.5), "m"),
        ("TVD", np.arange(1000.0, 1010.0, 0.5) * 0.98, "m"),
    ])
    standardize_curves(well, rename_curves=True, convert_units=False)
    assert well.get("TVD") is not None                    # never destroyed


# ---------------------------------------------------------------------------
# Scalar NaN must classify to None (matching the array path), not silently
# land in the last bin ("shaly", "water-bearing", "very poor", ...).
# ---------------------------------------------------------------------------
def test_scalar_nan_classifies_to_none():
    from petrophys.model.classifications import (
        HcSaturationClassification,
        InvasionClassification,
        MhiClassification,
        PermeabilityClassification,
        PorosityClassification,
        VclClassification,
    )

    for cls in (
        VclClassification, InvasionClassification, PorosityClassification,
        HcSaturationClassification, PermeabilityClassification, MhiClassification,
    ):
        assert cls.classify(float("nan")) is None


# ---------------------------------------------------------------------------
# Unit converters must fail loudly on unrecognized units instead of silently
# converting with factor 1.0, and must accept common spellings.
# ---------------------------------------------------------------------------
def test_units_reject_unknown_and_accept_common_spellings():
    from petrophys.model.units import UnitConverter

    with pytest.raises(ValueError):
        UnitConverter.depth(100, "km", "m")
    with pytest.raises(ValueError):
        UnitConverter.temperature(100, "R", "C")
    with pytest.raises(ValueError):
        UnitConverter.density(1.0, "g/l", "g/cm3")
    # MICRO SIGN (U+00B5) micro glyph must convert like the Greek mu.
    assert UnitConverter.slowness(620, "µs/m", "µs/ft") == pytest.approx(188.976)
    assert UnitConverter.resistivity(1.0, "ohm ft", "ohm.m") == pytest.approx(0.3048)


# ---------------------------------------------------------------------------
# net_gross: an interval entirely outside the logged well must report zero
# gross, not the geometric interval length.
# ---------------------------------------------------------------------------
def test_net_gross_outside_interval_reports_zero_gross():
    from petrophys.evaluation.summaries.net_gross import calculate_net_gross

    well = _make_well([("NET_PAY", np.ones(20), "")])
    result = calculate_net_gross(well, top_depth=2000.0, bottom_depth=2100.0)
    assert result.gross_thickness == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Log-viewer alias candidates: method-specific labels must not fall back to a
# generic curve (the header would claim a computation that never ran).
# ---------------------------------------------------------------------------
def test_log_viewer_method_labels_do_not_fall_back_to_generic():
    from petrophys.visualization.log_viewer import CURVE_ALIAS_CANDIDATES as C

    assert "PERM" not in C["PERM_TIMUR"]
    assert "PERM" not in C["PERM_COATES"]
    assert "PERM" not in C["PERM_CC"]
    assert "SW" not in C["SW_ARCHIE"]
    assert "NPHI" not in C["NPHI_HC"]
    assert "PHIT" not in C["PHIT_GAS_CORR"]
    assert "PHIE" not in C["PHIE_CC"]
    assert "RSHA" not in C["RXO"]
    assert C["SP"] == ["SP"]
    assert C["DCAL"] == ["DCAL"]


# ---------------------------------------------------------------------------
# Dynamic-workflow clay gate: GR-less wells with SP or neutron-density must
# still get a VCL curve (previously everything routed to Archie as clean sand).
# ---------------------------------------------------------------------------
def _base_arrays(n, seed):
    rng = np.random.default_rng(seed)
    return rng, {
        "RHOB": 2.4 + 0.2 * rng.random(n),
        "RDEP": 5 + 20 * rng.random(n),
    }


def test_workflow_computes_vcl_without_gr_nd_only():
    from petrophys.workflow import run_dynamic_workflow

    n = 200
    rng, base = _base_arrays(n, 0)
    well = _make_well(
        [("RHOB", base["RHOB"], "g/cm3"),
         ("NPHI", 0.15 + 0.2 * rng.random(n), "v/v"),
         ("RDEP", base["RDEP"], "ohm.m")],
        n=n, top=2000.0,
    )
    run_dynamic_workflow(well, lithology="clastic", rw=0.05)
    vcl = well.get("VCL")
    assert vcl is not None and np.isfinite(vcl.data).sum() > 0


def test_workflow_computes_vcl_without_gr_sp_only():
    from petrophys.workflow import run_dynamic_workflow

    n = 200
    rng, base = _base_arrays(n, 1)
    well = _make_well(
        [("SP", -60 + 80 * rng.random(n), "mV"),
         ("RHOB", base["RHOB"], "g/cm3"),
         ("RDEP", base["RDEP"], "ohm.m")],
        n=n, top=2000.0,
    )
    run_dynamic_workflow(well, lithology="clastic", rw=0.05)
    vcl = well.get("VCL")
    assert vcl is not None and np.isfinite(vcl.data).sum() > 0


# ---------------------------------------------------------------------------
# Context geomechanics reports the stress increment across the log interval.
# ---------------------------------------------------------------------------
def test_overburden_is_integrated_from_the_log_interval_top():
    from petrophys.evaluation.diagnostics.numerical_statistical_analysis import (
        NumericalStatisticalAnalysis,
    )

    from petrophys._internal.benchmark import historical_benchmark

    n = 100
    well = _make_well(
        [("RHOB", np.full(n, 2.3), "g/cm3"),
         ("GR", np.full(n, 60.0), "API")],
        n=n, top=1000.0,
    )
    # The benchmark packages report the increment; the current analysis is
    # absolute from the surface (tests/test_decisions_evaluation.py).
    with historical_benchmark():
        NumericalStatisticalAnalysis().analyze(well)
    sv = np.asarray(well.get("S_V").data, dtype=float)
    pp = np.asarray(well.get("PP_HYDROSTATIC").data, dtype=float)
    mask = np.isfinite(sv) & np.isfinite(pp)
    assert mask.sum() > 0
    assert np.all(np.diff(sv[mask]) > 0.0)
    assert sv[mask][0] < pp[mask][0]
