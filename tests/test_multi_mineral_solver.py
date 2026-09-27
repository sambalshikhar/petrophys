"""Tests for the multi-mineral solver, including the matrix-independence fix.

The solver used to subtract a pre-computed (matrix-dependent) PHIE from every
equation, which made the mineral result depend on the assumed matrix and created
a VCL<->lithology circularity. It now solves porosity as an unknown, so the
mineral solution is independent of any PHIE input.
"""

import numpy as np
import pytest

from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver


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
    # A porous quartz sand built as a self-consistent quartz + water mixture at
    # phi = 0.16, using the solver's own endpoints so the forward model and the
    # log readings agree on every channel (including U = PEF * rho, whose bulk
    # PEF here is ~1.70, not the 1.80 quartz-matrix value). The solver must then
    # invert it back to exactly quartz = 0.84, phi = 0.16.
    from petrophys.interpretation.lithology.multi_mineral_solver import _get_endpoints

    s = MultiMineralSolver()
    q, fl = _get_endpoints("quartz"), _get_endpoints("water")
    phi, vq = 0.16, 0.84
    rhob = np.array([vq * q["density"] + phi * fl["density"]])
    nphi = np.array([vq * q["nphi"] + phi * fl["nphi"]])
    u = vq * q["u"] + phi * fl["u"]              # U mixes volumetrically
    pef = np.array([u / rhob[0]])                # PEF = U / rho_bulk
    dt = np.array([vq * q["dt"] + phi * fl["dt"]])
    r = s.solve(rhob, nphi, np.zeros(1), np.zeros(1), pef=pef, dt=dt)
    assert r["phie"][0] == pytest.approx(0.16, abs=1e-3)
    assert r["quartz"][0] == pytest.approx(0.84, abs=1e-3)


def test_zero_nnls_result_with_remaining_rock_is_unsolved():
    result = MultiMineralSolver().solve(
        rhob=[0.5], nphi=[-0.1], vcl=[0.7], dt=[35.0], core_phi=[0.2],
    )
    assert not result["solved"][0]
    assert result["qc_flag"][0]
    assert not result["underdetermined"][0]
    for mineral in ("quartz", "calcite", "dolomite", "anhydrite"):
        assert np.isnan(result[mineral][0])


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_density_is_unsolved_without_aborting_other_rows(invalid):
    result = MultiMineralSolver().solve(
        rhob=[invalid, 2.71], nphi=[0.0, 0.0], vcl=0.0,
        pef=5.1, dt=47.6,
    )
    assert not result["solved"][0]
    assert result["solved"][1]
    assert result["calcite"][1] == pytest.approx(1.0, abs=0.01)


def test_duplicate_mineral_responses_are_not_independent_equations():
    result = MultiMineralSolver(minerals=["calcite", "calcite"]).solve(
        rhob=[2.71], nphi=[0.0], vcl=[0.0],
    )
    assert not result["solved"][0]
    assert result["qc_flag"][0]


@pytest.mark.parametrize('neutron,sonic', [(0.206, None), (None, 74.65)])
def test_reduced_carbonate_recovers_mixture_and_porosity(neutron, sonic):
    result = MultiMineralSolver(
        minerals_no_pef=["calcite", "dolomite"],
    ).solve(rhob=2.416, nphi=neutron, vcl=0.0, dt=sonic)
    assert result["solved"][0]
    assert result["calcite"][0] == pytest.approx(0.5, abs=1e-7)
    assert result["dolomite"][0] == pytest.approx(0.3, abs=1e-7)
    assert result["phie"][0] == pytest.approx(0.2, abs=1e-7)


def test_empty_solver_input_returns_empty_arrays():
    result = MultiMineralSolver().solve([], None, [])
    assert all(values.size == 0 for values in result.values())
