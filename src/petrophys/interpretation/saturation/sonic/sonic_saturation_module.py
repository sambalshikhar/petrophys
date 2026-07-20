"""Sonic (acoustic) water saturation via Gassmann fluid substitution.

An Sw estimate that is independent of resistivity / Rw -- useful in low-resistivity
low-contrast pay, shaly sands, and fresh-water zones where Archie breaks down.

Method (clean-room, public-domain rock physics):
  1. Vp from DTC (and Vs from DTS, else Greenberg-Castagna).
  2. Measured saturated moduli K_sat, mu from velocities + density.
  3. Dry-frame K_dry: calibrated from the well's water zone (preferred, self-
     consistent with the measured velocities) or a Krief frame model (fallback).
  4. Gassmann inverse -> in-situ fluid modulus K_fl.
  5. Sw from Wood mixing of brine and hydrocarbon moduli.

Applicability / limitations (validated, not marketing):
  * The math is exact -- on noise-free synthetic data with a known frame it
    recovers Sw to ~+/-0.02 when a measured shear log (DTS) is present.
  * The fluid effect on Vp is small (~2-3 %, ~2-3 us/ft) at typical porosity, so
    in REAL rock the dry-frame (K_dry) uncertainty and sonic noise easily swamp
    the saturation signal -- K_dry is the make-or-break input.
  * Best case: clean, high-porosity GAS sand with a measured DTS and a local
    (per-facies) frame calibration. Largely insensitive to OIL, and unreliable in
    shaly / heterogeneous intervals. It is a resistivity-INDEPENDENT cross-check,
    not a turnkey Sw -- treat it qualitatively unless locally calibrated.
  * Without DTS, Vs is estimated from Vp (Greenberg-Castagna), which absorbs the
    fluid signal and makes the result only indicative (flagged: vs_estimated).

Detectability gate (how the unreliability is made honest, not hidden):
  Rather than always returning a number, the method computes the brine->HC
  compressional-slowness swing per sample (the total fluid signal available) and
  reports Sw ONLY where that swing exceeds ``detect_factor`` x the sonic noise.
  Where the signal is below noise (stiff, low-porosity rock; oil/condensate),
  Sw is set to NaN and ``detectable`` is False, with a per-sample
  ``sw_uncertainty``. So the result is trustworthy where it is reported and
  abstains where the physics does not support an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from petrophys.interpretation.saturation.sonic.rock_physics import (
    MINERAL_ELASTIC, FLUID_ELASTIC,
    slowness_to_velocity, vs_from_vp_greenberg_castagna, velocity_to_moduli,
    dry_rock_modulus, gassmann_kfl, sw_from_fluid_modulus,
    fluid_substitution_dtc_swing,
)
from petrophys._internal.analysis_helpers import get_curve_data_depthwise


@dataclass
class SonicSaturationResult:
    sw: np.ndarray         # Sw, NaN where the fluid signal is below detectability
    k_sat: np.ndarray      # measured saturated bulk modulus (GPa)
    k_dry: np.ndarray      # dry-frame bulk modulus used (GPa)
    k_fl: np.ndarray       # inverted in-situ fluid modulus (GPa)
    vs_estimated: bool     # True if Vs was estimated from Vp (no DTS)
    kdry_method: str
    notes: str = ""
    sw_raw: Optional[np.ndarray] = None      # Sw before the detectability mask
    detectable: Optional[np.ndarray] = None  # True where fluid swing > detect_factor*noise
    sw_uncertainty: Optional[np.ndarray] = None  # +/- Sw from the sonic noise (clipped 0..1)
    dtc_fluid_swing: Optional[np.ndarray] = None  # brine->HC DTC swing (us/ft)


def _mineral_k_min(well, lithology: str, n: int) -> np.ndarray:
    """Voigt-Reuss-Hill mineral modulus per sample from VQUAR/VCALC/VDOLO/VCLAY,
    falling back to the matrix mineral implied by ``lithology``."""
    vol = {
        "quartz": get_curve_data_depthwise(well, ["VQUAR", "VSAND", "VQTZ"])[0],
        "calcite": get_curve_data_depthwise(well, ["VCALC", "VLIM"])[0],
        "dolomite": get_curve_data_depthwise(well, ["VDOLO", "VDOL"])[0],
        "clay": get_curve_data_depthwise(well, ["VCLAY", "VCL", "VSH"])[0],
    }
    have = {m: np.asarray(v, dtype=float) for m, v in vol.items() if v is not None}
    default = {"limestone": "calcite", "dolomite": "dolomite"}.get(lithology, "quartz")
    if not have:
        return np.full(n, MINERAL_ELASTIC[default]["k"])
    stack = np.vstack([have[m] for m in have])
    frac = np.where(np.isfinite(stack), np.clip(stack, 0.0, None), 0.0)
    tot = frac.sum(axis=0)
    k = np.full(n, MINERAL_ELASTIC[default]["k"], dtype=float)
    ok = tot > 1e-6
    ks = np.array([MINERAL_ELASTIC[m]["k"] for m in have])[:, None]
    f = frac / np.where(tot > 1e-6, tot, 1.0)
    voigt = (f * ks).sum(axis=0)
    reuss = 1.0 / np.where((f / ks).sum(axis=0) > 1e-9, (f / ks).sum(axis=0), np.nan)
    k[ok] = 0.5 * (voigt[ok] + reuss[ok])
    return k


def _calibrate_kdry(k_sat, phi, k_min, k_brine, water_mask):
    """Back out K_dry on water-bearing samples (K_fl=K_brine) and fit K_dry/K_min
    vs porosity, so the frame is consistent with the measured velocities.

    Uses the linearized Gassmann identity (Mavko & Mukerji):
        K_sat/(K_min-K_sat) = K_dry/(K_min-K_dry) + K_fl/(phi*(K_min-K_fl))
    -> L_dry = L_sat - L_fl, then K_dry = L_dry*K_min/(1+L_dry).
    """
    phi = np.clip(phi, 1e-3, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        l_sat = k_sat / (k_min - k_sat)
        l_fl = k_brine / (phi * (k_min - k_brine))
        l_dry = l_sat - l_fl
        k_dry_pts = l_dry * k_min / (1.0 + l_dry)
    f = k_dry_pts / k_min
    m = (water_mask & np.isfinite(f) & np.isfinite(phi) & (phi > 0) & (phi < 0.5)
         & (f > 0) & (f < 1) & (k_sat < k_min))
    if int(np.count_nonzero(m)) < 20:
        return None
    if np.ptp(phi[m]) < 1e-3:                       # ~constant porosity -> constant ratio
        f_all = np.full_like(phi, float(np.clip(np.median(f[m]), 0.02, 0.99)))
    else:
        a1, a0 = np.polyfit(phi[m], f[m], 1)        # linear K_dry/K_min trend
        f_all = np.clip(a0 + a1 * np.clip(phi, 0.0, 0.5), 0.02, 0.99)
    return k_min * f_all


def sonic_water_saturation(
    well,
    hc_type: str = "gas",
    lithology: str = "sandstone",
    k_brine: float = None,
    k_hc: float = None,
    kdry_method: str = "calibrated",
    water_zone_mask: Optional[np.ndarray] = None,
    sw_ref: Optional[Sequence[float]] = None,
    sw_water_cut: float = 0.85,
    frame_model: str = "krief",
    dtc_noise_us_ft: float = 2.0,
    detect_factor: float = 2.0,
) -> SonicSaturationResult:
    """Estimate water saturation from sonic + density via Gassmann.

    Parameters
    ----------
    hc_type : {"gas","oil"}        hydrocarbon end-member for fluid mixing.
    kdry_method : {"calibrated","model"}
        ``calibrated`` derives the dry frame from the well's water zone (needs
        ``water_zone_mask`` or ``sw_ref``); falls back to ``model`` if unavailable.
    sw_ref : array, optional        a reference Sw (e.g. Archie) used only to flag
        the water zone (Sw >= sw_water_cut) for K_dry calibration.
    dtc_noise_us_ft : float         assumed compressional-slowness noise (us/ft).
    detect_factor : float           Sw is reported only where the full brine->HC
        DTC swing exceeds ``detect_factor * dtc_noise_us_ft``; elsewhere the fluid
        signal is below the noise and Sw is set to NaN (``detectable`` is False).
        This is the honest scope of the method: it returns a number only where the
        sonic can actually resolve the fluid.
    """
    n = well.n_samples
    dt, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])
    dts, _, _ = get_curve_data_depthwise(well, ["DTS", "DTSM"])
    rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
    phi, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT", "PHIE_CC", "NPHI"])
    if dt is None or rhob is None or phi is None:
        raise ValueError("sonic saturation needs DTC, RHOB and a porosity curve")
    rhob = np.asarray(rhob, dtype=float)
    phi = np.clip(np.asarray(phi, dtype=float), 0.0, 0.5)

    vp = np.asarray(slowness_to_velocity(dt), dtype=float)
    vs_estimated = dts is None
    if dts is not None:
        vs = np.asarray(slowness_to_velocity(dts), dtype=float)
    else:
        vs = np.asarray(vs_from_vp_greenberg_castagna(vp, lithology), dtype=float)

    k_min = _mineral_k_min(well, lithology, n)
    k_sat, mu = velocity_to_moduli(vp, vs, rhob)

    kb = k_brine if k_brine is not None else FLUID_ELASTIC["brine"]["k"]
    khc = k_hc if k_hc is not None else FLUID_ELASTIC.get(hc_type, FLUID_ELASTIC["gas"])["k"]

    method = kdry_method
    k_dry = None
    if kdry_method == "calibrated":
        mask = None
        if water_zone_mask is not None:
            mask = np.asarray(water_zone_mask, dtype=bool)
        elif sw_ref is not None:
            mask = np.asarray(sw_ref, dtype=float) >= sw_water_cut
        if mask is not None:
            k_dry = _calibrate_kdry(k_sat, phi, k_min, kb, mask & np.isfinite(k_sat))
        if k_dry is None:
            method = "model (calibration unavailable)"
    if k_dry is None:
        k_dry, _ = dry_rock_modulus(phi, float(np.nanmedian(k_min)),
                                    MINERAL_ELASTIC["quartz"]["mu"], frame_model)
        k_dry = np.asarray(k_dry, dtype=float)

    k_fl = gassmann_kfl(k_sat, k_dry, k_min, phi)
    sw_raw = sw_from_fluid_modulus(k_fl, kb, khc)
    valid = np.isfinite(k_sat) & np.isfinite(k_dry) & (phi > 0.01)
    sw_raw = np.where(valid, sw_raw, np.nan)

    # Detectability: how much DTC actually moves between full brine and full HC.
    # If that swing is at/below the sonic noise, no inversion can resolve Sw, so
    # we report NaN there rather than a confident-looking but meaningless number.
    swing = fluid_substitution_dtc_swing(
        np.asarray(k_dry, dtype=float), k_min, mu, phi, rhob, kb, khc)
    with np.errstate(divide="ignore", invalid="ignore"):
        sw_uncertainty = np.clip(dtc_noise_us_ft / swing, 0.0, 1.0)
    detectable = valid & np.isfinite(swing) & (swing >= detect_factor * dtc_noise_us_ft)
    sw = np.where(detectable, sw_raw, np.nan)

    n_det = int(np.count_nonzero(detectable))
    n_val = int(np.count_nonzero(valid))
    frac = (n_det / n_val) if n_val else 0.0
    notes = ("Vs estimated from Vp (Greenberg-Castagna); result is approximate. "
             if vs_estimated else "Measured DTS used. ")
    notes += (f"Sw resolvable on {n_det}/{n_val} samples ({frac:.0%}): the brine->HC "
              f"DTC swing exceeds {detect_factor:g}x the {dtc_noise_us_ft:g} us/ft sonic "
              "noise only there; elsewhere Sw is NaN (fluid signal below noise).")
    return SonicSaturationResult(
        sw=sw, k_sat=k_sat, k_dry=np.asarray(k_dry, dtype=float),
        k_fl=k_fl, vs_estimated=vs_estimated, kdry_method=method, notes=notes,
        sw_raw=sw_raw, detectable=detectable, sw_uncertainty=sw_uncertainty,
        dtc_fluid_swing=swing)


class SonicSaturationModule:
    """Gassmann sonic-saturation calculator."""

    def __init__(self, hc_type: str = "gas", lithology: str = "sandstone",
                 kdry_method: str = "calibrated"):
        self.hc_type = hc_type
        self.lithology = lithology
        self.kdry_method = kdry_method

    def calculate(self, well, **kwargs) -> SonicSaturationResult:
        return sonic_water_saturation(
            well, hc_type=self.hc_type, lithology=self.lithology,
            kdry_method=self.kdry_method, **kwargs)

    def add_to_well(self, well, result: Optional[SonicSaturationResult] = None, **kwargs):
        if result is None:
            result = self.calculate(well, **kwargs)
        well.add_curve("SW_SONIC", result.sw, "v/v", "Sonic (Gassmann) water saturation")
        return well
