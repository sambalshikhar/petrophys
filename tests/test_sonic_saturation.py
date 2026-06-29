"""Tests for Gassmann sonic (acoustic) water saturation.

Validated against synthetic ground truth (forward-model Vp/Vs/rho from a known
Sw, then recover it): the method is exact with measured DTS and degrades to
approximate when Vs must be estimated from Vp (no shear log) -- a physical
limitation, flagged by the module.
"""

import numpy as np
import pytest

from petrophys import Well
from petrophys.petrophysics.sonic_saturation import (
    slowness_to_velocity, velocity_to_moduli, dry_rock_modulus,
    gassmann_ksat, gassmann_kfl, wood_fluid_modulus, sw_from_fluid_modulus,
    sonic_water_saturation, SonicSaturationModule,
)


# --- rock-physics primitives ---------------------------------------------------

def test_gassmann_fluid_substitution_round_trip():
    km, mum, phi = 37.0, 44.0, 0.25
    k_dry, _ = dry_rock_modulus(phi, km, mum, "krief")
    ks_b = gassmann_ksat(k_dry, km, 2.8, phi)
    ks_g = gassmann_ksat(k_dry, km, 0.06, phi)
    assert ks_g < ks_b                                    # gas softens the rock
    assert gassmann_kfl(ks_g, k_dry, km, phi) == pytest.approx(0.06, abs=1e-3)
    assert gassmann_kfl(ks_b, k_dry, km, phi) == pytest.approx(2.8, abs=1e-3)


def test_wood_and_sw_round_trip():
    kfl = wood_fluid_modulus(0.5, 2.8, 0.06)
    assert sw_from_fluid_modulus(kfl, 2.8, 0.06) == pytest.approx(0.5, abs=1e-6)
    assert sw_from_fluid_modulus(2.8, 2.8, 0.06) == pytest.approx(1.0)
    assert sw_from_fluid_modulus(0.06, 2.8, 0.06) == pytest.approx(0.0)


def test_slowness_velocity_and_moduli():
    vp = slowness_to_velocity(80.0)
    assert vp == pytest.approx(304800.0 / 80.0)
    k, mu = velocity_to_moduli(6008.0, 4000.0, 2.65)   # ~ quartz
    assert 80 < k + 4 / 3 * mu < 110 and mu > 0


def _synthetic_well(sw_true, phi, with_dts=True):
    km, mum, rhom = 37.0, 44.0, 2.65
    kdry, mudry = dry_rock_modulus(phi, km, mum, "krief")
    kfl = wood_fluid_modulus(sw_true, 2.8, 1.0)
    rho_fl = sw_true * 1.05 + (1 - sw_true) * 0.80
    rhob = (1 - phi) * rhom + phi * rho_fl
    ksat = gassmann_ksat(kdry, km, kfl, phi)
    vp = np.sqrt((ksat + 4 / 3 * mudry) / rhob * 1e6)
    vs = np.sqrt(mudry / rhob * 1e6)
    n = sw_true.size
    w = Well("SYN"); w.depth = np.arange(n, dtype=float)
    w.add_curve("DEPT", w.depth, "m", "d")
    w.add_curve("DTC", 304800.0 / vp, "us/ft", "dtc")
    if with_dts:
        w.add_curve("DTS", 304800.0 / vs, "us/ft", "dts")
    w.add_curve("RHOB", rhob, "g/cc", "rhob")
    w.add_curve("PHIE", phi, "v/v", "phi")
    return w


# --- recovery on synthetic ground truth ----------------------------------------

def test_recovers_known_sw_with_measured_dts():
    n = 400
    phi = np.full(n, 0.22) + 0.04 * np.sin(np.linspace(0, 8, n))
    sw_true = np.where(np.arange(n) < 240, 0.25, 1.0)   # HC top, water below
    w = _synthetic_well(sw_true, phi, with_dts=True)
    r = sonic_water_saturation(w, hc_type="oil", water_zone_mask=(sw_true >= 0.85))
    assert r.vs_estimated is False
    err = np.abs(r.sw - sw_true)
    assert np.nanmean(err) < 0.05                        # near-exact recovery
    assert np.nanmedian(r.sw[:240]) == pytest.approx(0.25, abs=0.1)
    assert np.nanmedian(r.sw[240:]) == pytest.approx(1.0, abs=0.1)


def test_flags_vs_estimated_without_dts():
    n = 200
    phi = np.full(n, 0.22)
    sw_true = np.where(np.arange(n) < 120, 0.3, 1.0)
    w = _synthetic_well(sw_true, phi, with_dts=False)
    r = sonic_water_saturation(w, hc_type="oil", water_zone_mask=(sw_true >= 0.85))
    assert r.vs_estimated is True                        # estimated from Vp
    assert "approximate" in r.notes.lower()


def test_requires_core_curves():
    w = Well("E"); w.depth = np.arange(10.0)
    w.add_curve("DEPT", w.depth, "m", "d")
    with pytest.raises(ValueError):
        sonic_water_saturation(w)


def test_module_adds_curve():
    n = 200
    phi = np.full(n, 0.2)
    sw_true = np.where(np.arange(n) < 100, 0.3, 1.0)
    w = _synthetic_well(sw_true, phi, with_dts=True)
    SonicSaturationModule(hc_type="oil").add_to_well(w, water_zone_mask=(sw_true >= 0.85))
    assert w.get("SW_SONIC") is not None
