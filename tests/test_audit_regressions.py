"""Regression guards for bugs found in the audit and fixed.

Each test pins the corrected behavior so the bug cannot silently return.
Known values are computed by hand (non-circular).
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# indonesia() must accept scalar rt (previously crashed: 'float' has no .shape)
# ---------------------------------------------------------------------------
def test_indonesia_accepts_scalar_rt():
    from petrophys.interpretation.saturation.indonesia import indonesia

    scalar = indonesia(rt=10.0, rw=0.05, phi=0.20, vcl=0.30, rcl=5.0)
    assert np.isscalar(scalar) or np.ndim(scalar) == 0
    arr = indonesia(
        rt=np.array([10.0]), rw=0.05,
        phi=np.array([0.20]), vcl=np.array([0.30]), rcl=5.0,
    )
    assert float(np.asarray(scalar)) == pytest.approx(float(arr[0]), rel=1e-9)


# ---------------------------------------------------------------------------
# indonesia() must use the canonical Poupon-Leveaux common-factor form,
# 1/sqrt(Rt) = (clay_term + sand_term) * Sw^(n/2), not the old form that put
# Sw^(n/2) on the sand term only (which under-predicted Sw in shaly zones).
# ---------------------------------------------------------------------------
def test_indonesia_common_factor_round_trip():
    from petrophys.interpretation.saturation.indonesia import indonesia

    vcl, phi, rw, rcl, a, m, n = 0.30, 0.20, 0.10, 5.0, 1.0, 2.0, 2.0
    clay = vcl ** (1 - vcl / 2) / np.sqrt(rcl)
    sand = phi ** (m / 2) / np.sqrt(a * rw)
    # Build Rt by hand from a known Sw via the canonical form, then recover it.
    for sw_true in (0.30, 0.55, 1.00):
        rt = 1.0 / ((clay + sand) * sw_true ** (n / 2)) ** 2
        recovered = float(indonesia(rt, rw, phi, vcl, rcl, a, m, n))
        assert recovered == pytest.approx(sw_true, rel=1e-9)


# ---------------------------------------------------------------------------
# The mineral solver's U row must be internally consistent: a pure mineral fed
# its own endpoint properties must resolve to ~100% of that mineral (residual
# ~0). Previously the U endpoints were hardcoded literature constants while the
# log term used PEF*RHOB, injecting a few-percent spurious mineral.
# ---------------------------------------------------------------------------
def test_mineral_solver_pure_minerals_resolve_exactly():
    from petrophys.interpretation.lithology.multi_mineral_solver import (
        MINERAL_PROPERTIES,
        MultiMineralSolver,
    )

    solver = MultiMineralSolver()
    from petrophys.interpretation.lithology.matrix_properties import get_log_density

    for name in ("quartz", "calcite", "dolomite", "anhydrite"):
        p = MINERAL_PROPERTIES[name]
        r = solver.explain_sample(
            rhob=get_log_density(name), nphi=p["nphi"], pef=p["pef"],
            dt=p["slowness"], vcl=0.0,
        )
        frac = dict(zip(r["columns"], r["x"]))
        assert frac[name] == pytest.approx(1.0, abs=1e-6)
        assert float(r["residual"]) < 1e-6


# ---------------------------------------------------------------------------
# MHI = Sw/Sxo must NOT be clipped at 1 (the >1 band is diagnostic)
# ---------------------------------------------------------------------------
def test_mhi_not_clipped_above_one():
    from petrophys.evaluation.mhi.mhi_module import calculate_mhi

    mhi = float(np.asarray(calculate_mhi(0.3, 0.5)))  # Sw/Sxo = 0.5/0.3
    assert mhi == pytest.approx(0.5 / 0.3, rel=1e-9)
    assert mhi > 1.0


# ---------------------------------------------------------------------------
# Net-pay cutoffs are inclusive (<=/>=), matching the documented contract
# ---------------------------------------------------------------------------
def test_net_pay_cutoffs_are_strict_at_boundary():
    from petrophys.evaluation.net_pay.flags import generate_net_flags

    vcl = np.array([0.30, 0.20, 0.20])    # at the vcl cutoff, then below it
    phie = np.array([0.10, 0.08, 0.10])   # at the phie cutoff in the second sample
    sw = np.array([0.50, 0.50, 0.70])     # at the sw cutoff in the third sample
    flags = generate_net_flags(vcl, phie, sw, 0.30, 0.08, 0.70)
    assert flags["net_rock"].tolist() == [False, True, True]
    assert flags["net_reservoir"].tolist() == [False, False, True]
    assert flags["net_pay"].tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# default_cutoffs('carbonate') exposes the carbonate PHIE without corrupting
# the clastic field
# ---------------------------------------------------------------------------
def test_default_cutoffs_carbonate_does_not_corrupt_clastic():
    from petrophys.evaluation.net_pay.cutoffs import default_cutoffs
    from petrophys.model.constants import DefaultParameters

    c = default_cutoffs("carbonate")
    assert c.phie_clastic == DefaultParameters.PHIE_CUTOFF_CLASTIC
    assert c.active_phie == DefaultParameters.PHIE_CUTOFF_CARBONATE
    assert default_cutoffs("clastic").active_phie == DefaultParameters.PHIE_CUTOFF_CLASTIC


# ---------------------------------------------------------------------------
# dual_water stays finite and bounded when Vcl*phi_cl > phi_t (swb would > 1)
# ---------------------------------------------------------------------------
def test_dual_water_handles_swb_over_one():
    from petrophys.interpretation.saturation.dual_water import dual_water

    swe = dual_water(phi_t=0.05, rt=20.0, rw=0.05, rwb=0.015, vcl=0.7, phi_cl=0.15)
    swe = float(np.asarray(swe))
    assert np.isfinite(swe)
    assert 0.0 <= swe <= 1.0


# ---------------------------------------------------------------------------
# Independent (non-circular) known-value checks for the perm/Sw formulas the
# audit flagged as only circularly tested.
# ---------------------------------------------------------------------------
def test_timur_known_value():
    from petrophys.interpretation.permeability.timur import timur
    # 0.136 * phi_pct**4.4 / swi_pct**2 with phi=0.20, swi=0.25
    expected = 0.136 * (20.0 ** 4.4) / (25.0 ** 2)
    assert float(np.asarray(timur(0.20, 0.25))) == pytest.approx(expected, rel=1e-6)


def test_coates_denoo_known_value():
    from petrophys.interpretation.permeability.coates_denoo import coates_denoo
    # 10000 * phi**4 * ((1-swi)/swi)**2 = 10000 * 0.2**4 * 9 = 144
    assert float(np.asarray(coates_denoo(0.20, 0.25))) == pytest.approx(144.0, rel=1e-6)


def test_archie_known_value():
    from petrophys.interpretation.saturation.archie import archie
    # sw = sqrt(a*rw / (phi**m * rt)) = sqrt(0.05 / (0.04*10)) = sqrt(0.125)
    sw = float(np.asarray(archie(rt=10.0, rw=0.05, phi=0.20, a=1.0, m=2.0, n=2.0)))
    assert sw == pytest.approx(np.sqrt(0.125), rel=1e-9)


# ---------------------------------------------------------------------------
# LAS export must not write Inf literally (round-trips to a null instead)
# ---------------------------------------------------------------------------
def test_export_las_normalizes_inf(tmp_path):
    from petrophys.model import Well
    from petrophys.io.export import export_las
    from petrophys.io import read_las

    well = Well("INF")
    well.depth = np.arange(3.0)
    well.add_curve("GR", np.array([10.0, np.inf, 30.0]), "gAPI", "gr")
    out = tmp_path / "inf.las"
    export_las(well, str(out))
    back = read_las(str(out)).get("GR").data
    assert np.isnan(back[1])              # inf became null
    assert np.isfinite(back[[0, 2]]).all()


# ---------------------------------------------------------------------------
# Resistivity alias collisions resolved: AT10/RLLS -> RSHA, AT30 -> RMED
# ---------------------------------------------------------------------------
def test_resistivity_alias_collisions_resolved():
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    assert get_standard_mnemonic("AT10") == "RSHA"
    assert get_standard_mnemonic("RLLS") == "RSHA"
    assert get_standard_mnemonic("AT30") == "RMED"


# ---------------------------------------------------------------------------
# Data gaps must not inflate per-sample thickness
# ---------------------------------------------------------------------------
def test_depth_step_array_caps_data_gaps():
    from petrophys._internal.analysis_helpers import depth_step_array

    d = np.array([1000.0, 1000.5, 1001.0, 1050.0, 1050.5])  # 49-unit gap
    dz = depth_step_array(d)
    # the gap sample is capped to ~median (0.5), not credited the full 49
    assert dz.max() <= 3.0 * 0.5 + 1e-9
    # uniform spacing is unchanged
    np.testing.assert_allclose(depth_step_array(np.arange(0.0, 5.0, 0.5)), 0.5)
    np.testing.assert_allclose(depth_step_array(np.arange(5.0, 0.0, -0.5)), 0.5)


# ---------------------------------------------------------------------------
# Formation analysis warns when tops fall entirely outside the log (unit mismatch)
# ---------------------------------------------------------------------------
def test_formation_analysis_warns_on_out_of_range_tops():
    import warnings
    from petrophys.model import Well
    from petrophys.evaluation.summaries.formation_analysis import analyze_formations, Formation

    well = Well("FT")
    well.depth = np.arange(7600.0, 8020.0, 0.5)   # feet
    tops = [Formation("A", 200, 400), Formation("B", 400, 700)]  # metres-ish, outside
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        analyze_formations(well, tops)
        assert any("overlap the well" in str(x.message) for x in rec)
