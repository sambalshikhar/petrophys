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
    from petrophys.petrophysics.saturation.indonesia import indonesia

    scalar = indonesia(rt=10.0, rw=0.05, phi=0.20, vcl=0.30, rcl=5.0)
    assert np.isscalar(scalar) or np.ndim(scalar) == 0
    arr = indonesia(
        rt=np.array([10.0]), rw=0.05,
        phi=np.array([0.20]), vcl=np.array([0.30]), rcl=5.0,
    )
    assert float(np.asarray(scalar)) == pytest.approx(float(arr[0]), rel=1e-9)


# ---------------------------------------------------------------------------
# MHI = Sw/Sxo must NOT be clipped at 1 (the >1 band is diagnostic)
# ---------------------------------------------------------------------------
def test_mhi_not_clipped_above_one():
    from petrophys.reservoir.mhi.mhi_module import calculate_mhi

    mhi = float(np.asarray(calculate_mhi(0.3, 0.5)))  # Sw/Sxo = 0.5/0.3
    assert mhi == pytest.approx(0.5 / 0.3, rel=1e-9)
    assert mhi > 1.0


# ---------------------------------------------------------------------------
# Net-pay cutoffs are inclusive (<=/>=), matching the documented contract
# ---------------------------------------------------------------------------
def test_net_pay_cutoffs_are_inclusive_at_boundary():
    from petrophys.reservoir.net_pay.flags import generate_net_flags

    vcl = np.array([0.30])    # exactly at vcl cutoff
    phie = np.array([0.08])   # exactly at phie cutoff
    sw = np.array([0.70])     # exactly at sw cutoff
    flags = generate_net_flags(vcl, phie, sw, 0.30, 0.08, 0.70)
    assert bool(flags["net_rock"][0])
    assert bool(flags["net_reservoir"][0])
    assert bool(flags["net_pay"][0])


# ---------------------------------------------------------------------------
# default_cutoffs('carbonate') exposes the carbonate PHIE without corrupting
# the clastic field
# ---------------------------------------------------------------------------
def test_default_cutoffs_carbonate_does_not_corrupt_clastic():
    from petrophys.reservoir.net_pay.cutoffs import default_cutoffs
    from petrophys.base.constants import DefaultParameters

    c = default_cutoffs("carbonate")
    assert c.phie_clastic == DefaultParameters.PHIE_CUTOFF_CLASTIC
    assert c.active_phie == DefaultParameters.PHIE_CUTOFF_CARBONATE
    assert default_cutoffs("clastic").active_phie == DefaultParameters.PHIE_CUTOFF_CLASTIC


# ---------------------------------------------------------------------------
# dual_water stays finite and bounded when Vcl*phi_cl > phi_t (swb would > 1)
# ---------------------------------------------------------------------------
def test_dual_water_handles_swb_over_one():
    from petrophys.petrophysics.saturation.dual_water import dual_water

    swe = dual_water(phi_t=0.05, rt=20.0, rw=0.05, rwb=0.015, vcl=0.7, phi_cl=0.15)
    swe = float(np.asarray(swe))
    assert np.isfinite(swe)
    assert 0.0 <= swe <= 1.0


# ---------------------------------------------------------------------------
# Independent (non-circular) known-value checks for the perm/Sw formulas the
# audit flagged as only circularly tested.
# ---------------------------------------------------------------------------
def test_timur_known_value():
    from petrophys.petrophysics.permeability.timur import timur
    # 0.136 * phi_pct**4.4 / swi_pct**2 with phi=0.20, swi=0.25
    expected = 0.136 * (20.0 ** 4.4) / (25.0 ** 2)
    assert float(np.asarray(timur(0.20, 0.25))) == pytest.approx(expected, rel=1e-6)


def test_coates_denoo_known_value():
    from petrophys.petrophysics.permeability.coates_denoo import coates_denoo
    # 10000 * phi**4 * ((1-swi)/swi)**2 = 10000 * 0.2**4 * 9 = 144
    assert float(np.asarray(coates_denoo(0.20, 0.25))) == pytest.approx(144.0, rel=1e-6)


def test_archie_known_value():
    from petrophys.petrophysics.saturation.archie import archie
    # sw = sqrt(a*rw / (phi**m * rt)) = sqrt(0.05 / (0.04*10)) = sqrt(0.125)
    sw = float(np.asarray(archie(rt=10.0, rw=0.05, phi=0.20, a=1.0, m=2.0, n=2.0)))
    assert sw == pytest.approx(np.sqrt(0.125), rel=1e-9)


# ---------------------------------------------------------------------------
# LAS export must not write Inf literally (round-trips to a null instead)
# ---------------------------------------------------------------------------
def test_export_las_normalizes_inf(tmp_path):
    from petrophys.base import Well
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
    from petrophys.qaqc.standardization.alias_map import get_standard_mnemonic

    assert get_standard_mnemonic("AT10") == "RSHA"
    assert get_standard_mnemonic("RLLS") == "RSHA"
    assert get_standard_mnemonic("AT30") == "RMED"


# ---------------------------------------------------------------------------
# Data gaps must not inflate per-sample thickness
# ---------------------------------------------------------------------------
def test_depth_step_array_caps_data_gaps():
    from petrophys.utils.analysis_helpers import depth_step_array

    d = np.array([1000.0, 1000.5, 1001.0, 1050.0, 1050.5])  # 49-unit gap
    dz = depth_step_array(d)
    # the gap sample is capped to ~median (0.5), not credited the full 49
    assert dz.max() <= 3.0 * 0.5 + 1e-9
    # uniform spacing is unchanged
    np.testing.assert_allclose(depth_step_array(np.arange(0.0, 5.0, 0.5)), 0.5)


# ---------------------------------------------------------------------------
# Formation analysis warns when tops fall entirely outside the log (unit mismatch)
# ---------------------------------------------------------------------------
def test_formation_analysis_warns_on_out_of_range_tops():
    import warnings
    from petrophys.base import Well
    from petrophys.summation.formation_analysis import analyze_formations, Formation

    well = Well("FT")
    well.depth = np.arange(7600.0, 8020.0, 0.5)   # feet
    tops = [Formation("A", 200, 400), Formation("B", 400, 700)]  # metres-ish, outside
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        analyze_formations(well, tops)
        assert any("overlap the well" in str(x.message) for x in rec)
