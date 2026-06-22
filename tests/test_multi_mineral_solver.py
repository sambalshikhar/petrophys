"""Tests for the multi-mineral solver, including the matrix-independence fix.

The solver used to subtract a pre-computed (matrix-dependent) PHIE from every
equation, which made the mineral result depend on the assumed matrix and created
a VCL<->lithology circularity. It now solves porosity as an unknown, so the
mineral solution is independent of any PHIE input.
"""

import numpy as np
import pytest

from petrophys.lithology.multi_mineral_solver import MultiMineralSolver


# clean (zero-porosity) matrix points with a full log set
RHOB = np.array([2.65, 2.71, 2.87])
NPHI = np.array([-0.02, 0.00, 0.02])
PEF = np.array([1.8, 5.1, 3.1])
DT = np.array([55.5, 47.6, 43.5])
VCL = np.zeros(3)


def test_solver_recovers_clean_minerals():
    s = MultiMineralSolver()
    r = s.solve(RHOB, NPHI, VCL, np.zeros(3), pef=PEF, dt=DT)
    assert r["quartz"][0] == pytest.approx(1.0, abs=0.05)
    assert r["calcite"][1] == pytest.approx(1.0, abs=0.1)
    assert r["dolomite"][2] == pytest.approx(1.0, abs=0.05)
    # zero-porosity matrix points -> ~0 solved porosity
    assert np.all(r["phie"] < 0.05)


def test_solver_mineral_result_is_independent_of_phie_input():
    # The core fix: identical logs + Vcl but different PHIE inputs must give the
    # same mineral volumes (porosity is solved, not subtracted as a hard input).
    s = MultiMineralSolver()
    rhob = np.array([2.55, 2.68, 2.80, 2.62])
    nphi = np.array([0.20, 0.10, 0.05, 0.15])
    pef = np.array([2.0, 4.5, 3.0, 2.2])
    dt = np.array([80.0, 65.0, 60.0, 75.0])
    vcl = np.array([0.10, 0.10, 0.10, 0.10])

    r_low = s.solve(rhob, nphi, vcl, np.full(4, 0.02), pef=pef, dt=dt)
    r_high = s.solve(rhob, nphi, vcl, np.full(4, 0.35), pef=pef, dt=dt)

    for mineral in ("quartz", "calcite", "dolomite", "anhydrite"):
        np.testing.assert_allclose(r_low[mineral], r_high[mineral], atol=1e-9)


def test_core_porosity_constraint_pins_phi_and_resolves_carbonate():
    # Self-consistent porous limestone, NO PEF. With core porosity pinned, the
    # matrix density is determined and the solver resolves calcite over quartz.
    s = MultiMineralSolver()
    phi = 0.20
    rhob = np.array([2.71 * (1 - phi) + 1.0 * phi])
    nphi = np.array([0.0 * (1 - phi) + 1.0 * phi])
    dt = np.array([47.6 * (1 - phi) + 189.0 * phi])
    r = s.solve(rhob, nphi, np.zeros(1), np.zeros(1),
                pef=None, dt=dt, core_phi=np.array([phi]))
    assert r["calcite"][0] > r["quartz"][0]
    assert r["phie"][0] == pytest.approx(phi, abs=1e-6)   # porosity pinned to core


def test_solver_porosity_is_solved_and_physical():
    # A porous quartz sand: rhob between matrix and fluid -> positive solved phi.
    s = MultiMineralSolver()
    rhob = np.array([2.39])      # ~0.16 porosity quartz
    nphi = np.array([0.16])
    pef = np.array([1.8])
    dt = np.array([75.0])
    vcl = np.zeros(1)
    r = s.solve(rhob, nphi, vcl, np.zeros(1), pef=pef, dt=dt)
    assert r["phie"][0] == pytest.approx(0.16, abs=0.05)
    assert r["quartz"][0] == pytest.approx(0.84, abs=0.06)
