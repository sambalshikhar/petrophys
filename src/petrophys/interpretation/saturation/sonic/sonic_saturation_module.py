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

Applicability / limitations:
  * The inversion is exact: noise-free synthetic data with a measured shear
    log (DTS) and the dry frame that generated them return the input Sw. A
    frame calibrated from the water zone is a linear K_dry/K_min trend in
    porosity and carries its own error away from the calibrated porosities.
  * The fluid effect on Vp is small (~2-3 %, ~2-3 us/ft) at typical porosity, so
    in REAL rock the dry-frame (K_dry) uncertainty and sonic noise easily swamp
    the saturation signal -- K_dry is the make-or-break input.
  * Best case: clean, high-porosity GAS sand with a measured DTS and a local
    (per-facies) frame calibration. Largely insensitive to OIL, and unreliable in
    shaly / heterogeneous intervals. It is a resistivity-INDEPENDENT cross-check,
    not a turnkey Sw -- treat it qualitatively unless locally calibrated.
  * Without DTS, Vs is estimated from Vp (Greenberg-Castagna), which absorbs the
    fluid signal and makes the result only indicative (flagged: vs_estimated).

Detectability gate:
  Rather than always returning a number, the method computes the brine->HC
  compressional-slowness swing per sample (the total fluid signal available) and
  reports Sw ONLY where that swing exceeds ``detect_factor`` x the sonic noise.
  Where the signal is below noise (stiff, low-porosity rock; oil/condensate),
  Sw is set to NaN and ``detectable`` is False, with a per-sample
  ``sw_uncertainty``. The method abstains where the fluid signal is below the
  sonic noise. The gate does not include the error of the dry-frame modulus
  (K_dry): with the default calibrated frame a few percent K_dry error can move
  a reported gas Sw by several tenths, so a reported Sw is not error-free.

References
----------
Gassmann, F. (1951). Über die Elastizität poröser Medien.
*Vierteljahrsschrift der Naturforschenden Gesellschaft in Zürich* 96, 1-23.

Greenberg, M.L. and Castagna, J.P. (1992). Shear-wave velocity estimation
in porous rocks: theoretical formulation, preliminary verification and
applications. *Geophysical Prospecting* 40, 195-209.

Mavko, G., Mukerji, T. and Dvorkin, J. (2009). *The Rock Physics
Handbook*, 2nd ed. Cambridge University Press.

Wood, A.B. (1955). *A Textbook of Sound*, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from petrophys.interpretation.saturation.sonic.rock_physics import (
    MINERAL_ELASTIC, FLUID_ELASTIC, _GC,
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
    """Voigt-Reuss-Hill mineral modulus (GPa) per sample from the mineral
    volumes on the well (quartz, calcite, dolomite and clay).

    The matrix mineral is the one implied by ``lithology`` (calcite for
    limestone, dolomite for dolomite, else quartz). Where its own volume is
    logged the volumes are scaled to sum to 1. Where it is not (a well with
    a clay volume alone), the matrix mineral fills what the logged volumes
    leave of 1, so clean rock has the modulus of the matrix mineral. A well
    with no mineral volume has the matrix mineral throughout.
    """
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
    matrix_logged = np.isfinite(have[default]) if default in have else np.zeros(n, dtype=bool)
    have.setdefault(default, np.zeros(n))
    stack = np.vstack([have[m] for m in have])
    frac = np.where(np.isfinite(stack), np.clip(stack, 0.0, None), 0.0)
    # The matrix mineral makes up the rest of the solid where it is not logged.
    rest = np.where(matrix_logged, 0.0, np.clip(1.0 - frac.sum(axis=0), 0.0, None))
    frac[list(have).index(default)] += rest
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
    well : Well
        Well with a compressional slowness (DTC or DT, us/ft), a bulk
        density (RHOB) and a porosity curve (the first of PHIE, PHIT,
        PHIE_CC and NPHI with a value at each depth). A shear slowness (DTS)
        and mineral volumes are used where present.
    hc_type : {"gas","oil"}
        Hydrocarbon end-member for fluid mixing.
    lithology : {"sandstone","limestone","dolomite","shale"}
        Greenberg-Castagna Vs relation (without DTS) and the matrix mineral:
        calcite for limestone, dolomite for dolomite, else quartz. The
        mineral modulus is the Voigt-Reuss-Hill average of the mineral
        volumes on the well; a well with a clay volume and no volume of the
        matrix mineral has the matrix mineral as the rest of the solid.
    k_brine : float, optional
        Bulk modulus of the brine, GPa; default 2.8.
    k_hc : float, optional
        Bulk modulus of the hydrocarbon, GPa; default 0.06 for gas and 1.0
        for oil.
    kdry_method : {"calibrated","model"}
        ``calibrated`` derives the dry frame from the well's water zone (needs
        ``water_zone_mask`` or ``sw_ref``); falls back to ``model`` if unavailable.
    water_zone_mask : array of bool, optional
        True at the water-bearing samples the dry frame is calibrated on;
        used in place of ``sw_ref``. At least 20 usable samples are needed.
    sw_ref : array, optional
        A reference Sw (e.g. Archie) used only to flag the water zone
        (Sw >= sw_water_cut) for K_dry calibration.
    sw_water_cut : float, default 0.85
        ``sw_ref`` at or above this value marks a water-zone sample.
    frame_model : {"krief","critical_porosity"}
        Dry-frame model used when the frame is not calibrated, with each
        sample's mineral modulus.
    dtc_noise_us_ft : float
        Assumed compressional-slowness noise (us/ft).
    detect_factor : float
        Sw is reported only where the full brine->HC DTC swing exceeds
        ``detect_factor * dtc_noise_us_ft``; elsewhere the fluid signal is
        below the noise and Sw is set to NaN (``detectable`` is False). This
        is the honest scope of the method: it returns a number only where the
        sonic can actually resolve the fluid.

    Returns
    -------
    SonicSaturationResult
        ``sw`` (v/v, NaN where not detectable), ``sw_raw``, ``detectable``,
        ``sw_uncertainty``, ``dtc_fluid_swing`` (us/ft), the moduli ``k_sat``,
        ``k_dry`` and ``k_fl`` (GPa), ``vs_estimated``, ``kdry_method`` and
        ``notes``.

    Raises
    ------
    ValueError
        If ``hc_type``, ``lithology`` or ``kdry_method`` is not one of its
        choices, or the well lacks DTC, RHOB or a porosity curve.
    """
    choices = {
        "hc_type": (hc_type, sorted(set(FLUID_ELASTIC) - {"brine"})),
        "lithology": (lithology, sorted(_GC)),
        "kdry_method": (kdry_method, ["calibrated", "model"]),
    }
    for name, (value, allowed) in choices.items():
        if str(value).strip().lower() not in allowed:
            raise ValueError(f"{name} must be one of {allowed}; got {value!r}")
    hc_type = str(hc_type).strip().lower()
    lithology = str(lithology).strip().lower()
    kdry_method = str(kdry_method).strip().lower()
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
        # The frame has each sample's mineral modulus, as the Gassmann inversion has.
        k_dry, _ = dry_rock_modulus(phi, k_min, MINERAL_ELASTIC["quartz"]["mu"], frame_model)
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
        """Estimate the well's sonic water saturation.

        Parameters
        ----------
        well : Well
            Well with DTC, RHOB and a porosity curve.
        **kwargs
            Other options of :func:`sonic_water_saturation`, e.g. ``sw_ref``
            or ``water_zone_mask`` for the dry-frame calibration.

        Returns
        -------
        SonicSaturationResult
            The result of :func:`sonic_water_saturation` with this module's
            ``hc_type``, ``lithology`` and ``kdry_method``.
        """
        return sonic_water_saturation(
            well, hc_type=self.hc_type, lithology=self.lithology,
            kdry_method=self.kdry_method, **kwargs)

    def add_to_well(self, well, result: Optional[SonicSaturationResult] = None, **kwargs):
        """Add the sonic water saturation to the well as ``SW_SONIC``.

        Parameters
        ----------
        well : Well
            Well to add the curve to.
        result : SonicSaturationResult, optional
            Result of :meth:`calculate` for this well; calculated when
            omitted.
        **kwargs
            Options of :meth:`calculate`, used when ``result`` is omitted.

        Returns
        -------
        Well
            ``well``, with ``SW_SONIC`` (v/v; NaN where the fluid signal is
            below the sonic noise) added or replaced.
        """
        if result is None:
            result = self.calculate(well, **kwargs)
        well.add_curve("SW_SONIC", result.sw, "v/v", "Sonic (Gassmann) water saturation")
        return well
