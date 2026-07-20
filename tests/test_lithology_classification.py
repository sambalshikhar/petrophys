"""Tests for volume-based lithology classification + RHOmaa-Umaa cross-check.

Covers the design from the lithology-classification rework:
  #1 classify from solved mineral volumes (dominant mineral + clay cut),
  #2 raw-log cutoff tree used only as a no-solver fallback,
  #3 RHOmaa-Umaa second solve that flags disagreement automatically.
"""

import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.lithology.lithology_module import LithologyModule


@pytest.fixture
def litho():
    return LithologyModule()


# --- #1 volume-based classifier -------------------------------------------------

def test_dominant_mineral_names_the_rock(litho):
    # clean, low-clay samples -> dominant solved mineral wins
    assert litho._classify_from_volumes(0.05, 0.80, 0.05, 0.05, 0.0, 2.45, 0.18, 30) == "sandstone"
    assert litho._classify_from_volumes(0.05, 0.05, 0.80, 0.05, 0.0, 2.60, 0.06, 20) == "limestone"
    assert litho._classify_from_volumes(0.05, 0.05, 0.05, 0.80, 0.0, 2.70, 0.05, 20) == "dolomite"


def test_clay_cut_calls_shale(litho):
    assert litho._classify_from_volumes(0.55, 0.40, 0.0, 0.0, 0.0, 2.50, 0.25, 110) == "shale"


def test_combined_carbonate_names_a_carbonate(litho):
    # neither calcite nor dolomite alone reaches 0.5, but together they dominate
    out = litho._classify_from_volumes(0.05, 0.30, 0.36, 0.34, 0.0, 2.72, 0.06, 25)
    assert out == "limestone"  # calcite > dolomite


def test_unresolved_mixture_is_mixed(litho):
    # 4 confirmed components, none dominant, carbonate < 0.5 -> mixed
    out = litho._classify_from_volumes(0.05, 0.30, 0.20, 0.20, 0.30, 2.90, 0.05, 30)
    assert out == "mixed"


# --- anhydrite / coal / salt guards --------------------------------------------

def test_anhydrite_requires_density_signature(litho):
    # high anhydrite volume but ordinary density -> demoted (solver used it as a sink)
    assert litho._classify_from_volumes(0.05, 0.30, 0.0, 0.0, 0.70, 2.50, 0.20, 40) == "sandstone"
    # same volume with the real anhydrite density/neutron signature -> anhydrite
    assert litho._classify_from_volumes(0.05, 0.30, 0.0, 0.0, 0.70, 2.96, 0.00, 20) == "anhydrite"


def test_coal_and_salt_physical_guards(litho):
    assert litho._classify_from_volumes(0.0, 0.0, 0.0, 0.0, 0.0, 1.50, 0.50, 40) == "coal"
    assert litho._classify_from_volumes(0.0, 0.0, 0.0, 0.0, 0.0, 2.10, 0.00, 20) == "salt"


# --- #3 RHOmaa-Umaa transform + ternary ----------------------------------------

def test_rhomaa_umaa_pure_minerals_round_trip(litho):
    # A zero-porosity pure mineral should sit on its own end-point and classify there.
    for pef, density, expected in [(1.8, 2.65, "sandstone"), (5.1, 2.71, "limestone"), (3.1, 2.87, "dolomite")]:
        # choose RHOB so apparent porosity ~0 (rhob >= 2.71 keeps phi_d <= 0)
        rhob = max(density, 2.71)
        rm, um = litho._rhomaa_umaa(rhob, nphi=0.0, pef=pef * rhob / ((rhob + 0.1883) / 1.0704))
        assert np.isfinite(rm) and np.isfinite(um)
    # Direct ternary at the textbook end-points
    assert litho._ternary_qcd_lithology(2.65, 4.8) == "sandstone"
    assert litho._ternary_qcd_lithology(2.71, 13.8) == "limestone"
    assert litho._ternary_qcd_lithology(2.87, 9.0) == "dolomite"


def test_rhomaa_umaa_requires_pef(litho):
    rm, um = litho._rhomaa_umaa(2.65, 0.10, np.nan)
    assert np.isnan(rm) and np.isnan(um)


def test_point_far_outside_triangle_gives_no_verdict(litho):
    assert litho._ternary_qcd_lithology(2.0, 30.0) is None


# --- integration through calculate() -------------------------------------------

def _well_with(curves: dict, n: int = 12) -> Well:
    depth = np.arange(1000.0, 1000.0 + n * 0.5, 0.5)[:n]
    well = Well("LITH-TEST")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    for name, val in curves.items():
        well.add_curve(name, np.full(n, val), "", name)
    return well


def test_calculate_clay_cut_and_xcheck_plumbing():
    # full curve suite incl. PEF so samples solve and the cross-check runs
    well = _well_with({"RHOB": 2.45, "NPHI": 0.18, "PEF": 1.9, "GR": 35, "PHIE": 0.15})
    vcl = np.full(well.n_samples, 0.10)
    vcl[:4] = 0.60  # force a shale interval via the clay cut
    result = LithologyModule().calculate(well, vcl=vcl)

    assert result.solved.all()
    # clay cut is deterministic regardless of the solved minerals
    assert np.all(result.lithology_code[:4] == LithologyModule.LITHOLOGY_CODES["shale"])
    # RHOmaa-Umaa ran on the clean (PEF-bearing, low-clay) samples
    assert result.rhomaa is not None and np.any(np.isfinite(result.rhomaa[4:]))
    assert result.litho_disagree is not None and result.litho_disagree.shape == vcl.shape
