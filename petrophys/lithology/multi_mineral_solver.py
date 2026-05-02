"""
Multi-mineral solver for lithology determination.

This module provides solvers for determining mineral fractions
from multiple log measurements using simultaneous equations.
"""

from typing import Optional, Tuple, Dict, Union, Sequence
import numpy as np
from scipy.optimize import nnls
from petrophys.base.well import Well

from petrophys.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)
from petrophys.utils.analysis_helpers import get_curve_data_depthwise


def _u(pef: float, density: float) -> float:
    """Volumetric photoelectric cross-section: U = PEF × ρ [b/cm³]."""
    return pef * density


def _get_endpoints(name: str) -> Dict[str, float]:
    """Return endpoint dict with density, nphi, pef, u, dt for a mineral/fluid."""
    p = get_mineral_properties(name)
    return {
        "density": p["density"],
        "nphi":    p["nphi"],
        "pef":     p["pef"],
        "u":       p["u"],
        "dt":      p["slowness"],
    }


def _constrain_mineral_fractions(
    fractions: np.ndarray,
    available_fraction: float,
) -> np.ndarray:
    """Clip mineral fractions and enforce closure to the available rock volume."""
    constrained = np.asarray(fractions, dtype=float).copy()
    constrained = np.clip(constrained, 0.0, max(float(available_fraction), 0.0))
    total = float(np.nansum(constrained))
    if total > available_fraction and total > 0.0:
        constrained *= float(available_fraction) / total
    return constrained


class MultiMineralSolver:
    """
    Multi-mineral solver for complex lithology.

    This class solves for mineral fractions using multiple
    log measurements simultaneously.

    Vcl (from VclModule) and PHIE (from PorosityModule) must be
    pre-computed before calling this solver. Both are subtracted
    from the RHS of every equation before solving.

    The active mineral set and equations are determined per depth
    based on curve availability:

        PEF available              → 4 minerals (quartz, calcite, dolomite,
                                     anhydrite) + PEF row (+ DTC if present)
        PEF absent, DTC available  → 4 minerals (quartz, calcite, dolomite,
                                     anhydrite) + DTC row
        Neither available          → 3 minerals, RHOB + NPHI only

    PEF is preferred over DTC for mineral discrimination because U = PEF × ρ
    has a much wider spread across carbonates (quartz: 4.77, dolomite: 8.90,
    calcite: 13.82 b/cm³) and is the only tool that reliably separates
    anhydrite from dolomite.

    Tool Response Equations (linear mixing):
        Unity:   V1 + V2 + ... + Vn         = 1    - Vcl - φe
        Density: V1·ρ1 + V2·ρ2 + ... + Vn·ρn = ρlog - Vcl·ρcl - φe·ρfl
        Neutron: V1·N1 + V2·N2 + ... + Vn·Nn = Nlog - Vcl·Ncl - φe·Nfl
        U:       V1·U1 + V2·U2 + ... + Vn·Un = Ulog - Vcl·Ucl - φe·Ufl
        Sonic:   V1·D1 + V2·D2 + ... + Vn·Dn = Dlog - Vcl·Dcl - φe·Dfl

    where U = PEF × ρ (volumetric photoelectric cross-section).

    Parameters
    ----------
    minerals : list, optional
        List of mineral types to solve for when PEF is available.
        Default includes anhydrite as the 4th mineral:
        ["quartz", "calcite", "dolomite", "anhydrite"]
    minerals_no_pef : list, optional
        Mineral list used when PEF is absent (default: first three
        from minerals, i.e. ["quartz", "calcite", "dolomite"])
    clay_mineral : str, optional
        Clay mineral name for endpoint lookup (default: "clay")
    fluid : str, optional
        Fluid type (default: "water")

    Examples
    --------
    >>> solver = MultiMineralSolver()
    >>> result = solver.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)
    """

    def __init__(
        self,
        minerals: list = None,
        minerals_no_pef: list = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]
        if minerals_no_pef is None:
            minerals_no_pef = minerals[:3]

        self.minerals        = minerals
        self.minerals_no_pef = minerals_no_pef
        self.clay_mineral    = clay_mineral
        self.fluid           = fluid

        self._load_properties()

    def _load_properties(self):
        """Load mineral and fluid properties."""
        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))
        self.mineral_props = {m: _get_endpoints(m) for m in all_minerals}
        self.fluid_props   = _get_endpoints(self.fluid)
        self.clay_props    = _get_endpoints(self.clay_mineral)

    def _solve_sample(
        self,
        rhob: float,
        nphi: float,
        vcl: float,
        phie: float,
        pef: Optional[float],
        dt: Optional[float],
    ) -> Tuple[str, ...]:
        """
        Solve for mineral fractions at a single depth sample.

        Active mineral set is chosen based on available curves:
            PEF present → self.minerals (4 minerals)
            PEF absent, DTC present → self.minerals (4 minerals) + DTC row
            Neither → self.minerals_no_pef (3 minerals, RHOB+NPHI only)

        Parameters
        ----------
        rhob : float
            Bulk density (g/cm³)
        nphi : float
            Neutron porosity (v/v)
        vcl : float
            Clay volume fraction (0-1)
        phie : float
            Effective porosity (0-1)
        pef : float or None
            Photoelectric factor (b/e)
        dt : float or None
            Sonic slowness (μs/ft)

        Returns
        -------
        tuple
            (active_minerals, fractions) where active_minerals is the
            list of mineral names used and fractions are the solved volumes
        """
        vcl  = float(np.clip(vcl,  0.0, 1.0))
        phie = float(np.clip(phie, 0.0, 1.0))

        # Select active mineral set based on curve availability.
        # 4 minerals need at least 4 equations: Unity + RHOB + NPHI + (PEF or DTC).
        # Drop to 3 minerals only when neither is available,
        # leaving exactly 3 equations for 3 unknowns.
        has_discriminator = pef is not None or dt is not None
        active_minerals = self.minerals if has_discriminator else self.minerals_no_pef

        m  = [self.mineral_props[name] for name in active_minerals]
        fl = self.fluid_props
        cl = self.clay_props

        A_rows, b_rows = [], []

        def add_row(mineral_vals, cl_val, fl_val, log_val):
            rhs = log_val - vcl * cl_val - phie * fl_val
            A_rows.append(mineral_vals)
            b_rows.append(rhs)

        # Unity  (cl and fl endpoints are both 1)
        add_row([1.0] * len(active_minerals),
                cl_val=1.0,           fl_val=1.0,           log_val=1.0)

        # Density
        add_row([mi["density"] for mi in m],
                cl_val=cl["density"], fl_val=fl["density"], log_val=rhob)

        # Neutron
        add_row([mi["nphi"] for mi in m],
                cl_val=cl["nphi"],    fl_val=fl["nphi"],    log_val=nphi)

        # U = PEF × ρ
        if pef is not None:
            add_row([mi["u"] for mi in m],
                    cl_val=cl["u"],   fl_val=fl["u"],   log_val=_u(pef, rhob))

        # Sonic (DTC) — used when PEF absent, or as extra row when both present
        if dt is not None:
            add_row([mi["dt"] for mi in m],
                    cl_val=cl["dt"],  fl_val=fl["dt"],  log_val=dt)

        x, _ = nnls(np.array(A_rows), np.array(b_rows))

        # Check closure against available mineral space (1 - Vcl - φe)
        available_fraction = 1.0 - vcl - phie
        if abs(x.sum() - available_fraction) > 0.05:
            return active_minerals, tuple(np.nan for _ in active_minerals)
        x = _constrain_mineral_fractions(x, available_fraction)

        return active_minerals, tuple(float(v) for v in x)

    def solve(
        self,
        rhob: Union[float, np.ndarray],
        nphi: Union[float, np.ndarray],
        vcl:  Union[float, np.ndarray],
        phie: Union[float, np.ndarray],
        pef:  Optional[Union[float, np.ndarray]] = None,
        dt:   Optional[Union[float, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve for mineral fractions.

        Parameters
        ----------
        rhob : float or np.ndarray
            Bulk density (g/cm³)
        nphi : float or np.ndarray
            Neutron porosity (v/v)
        vcl : float or np.ndarray
            Pre-computed clay volume (0-1)
        phie : float or np.ndarray
            Pre-computed effective porosity (0-1)
        pef : float or np.ndarray, optional
            Photoelectric factor (b/e) — omit if curve absent
        dt : float or np.ndarray, optional
            Sonic slowness (μs/ft) — omit if curve absent

        Returns
        -------
        dict
            Dictionary with mineral fractions, vcl, phie, and qc_flag.
            All minerals from both self.minerals and self.minerals_no_pef
            are present as keys — depths where a mineral was not in the
            active set are filled with 0.
        """
        rhob = np.atleast_1d(np.asarray(rhob, dtype=float))
        nphi = np.atleast_1d(np.asarray(nphi, dtype=float))
        vcl  = np.atleast_1d(np.asarray(vcl,  dtype=float))
        phie = np.atleast_1d(np.asarray(phie, dtype=float))
        n    = len(rhob)

        pef_arr = np.atleast_1d(np.asarray(pef, dtype=float)) if pef is not None else None
        dt_arr  = np.atleast_1d(np.asarray(dt,  dtype=float)) if dt  is not None else None

        # All possible minerals across both sets
        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))

        result = {mineral: np.zeros(n) for mineral in all_minerals}
        result["vcl"]     = vcl.copy()
        result["phie"]    = phie.copy()
        result["qc_flag"] = np.zeros(n, dtype=bool)

        for i in range(n):
            if any(np.isnan([rhob[i], nphi[i], vcl[i], phie[i]])):
                for mineral in all_minerals:
                    result[mineral][i] = np.nan
                result["qc_flag"][i] = True
                continue

            pef_i = float(pef_arr[i]) if pef_arr is not None else None
            dt_i  = float(dt_arr[i])  if dt_arr  is not None else None

            if pef_i is not None and np.isnan(pef_i):
                pef_i = None
            if dt_i  is not None and np.isnan(dt_i):
                dt_i  = None

            active_minerals, fractions = self._solve_sample(
                rhob[i], nphi[i], vcl[i], phie[i], pef_i, dt_i
            )

            for mineral, v in zip(active_minerals, fractions):
                result[mineral][i] = v

            if any(np.isnan(fractions)):
                result["qc_flag"][i] = True

        return result

    def add_to_well(self, well: "Well") -> "Well":
        """
        Add lithology curves to well.

        Requires RHOB, NPHI, VCL, and PHIE on the well object.
        VCL must have been added by VclModule and PHIE by
        PorosityModule before calling this. PEF and DTC are
        read if present and determine the active mineral set.

        Parameters
        ----------
        well : Well
            Well object

        Returns
        -------
        Well
            Well with lithology curves added
        """
        rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
        nphi, _, _ = get_curve_data_depthwise(well, ["NPHI", "PHIN"])
        vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        pef, _, _ = get_curve_data_depthwise(well, ["PEF"])
        dt, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])

        missing = [
            name for name, data in zip(
                ["RHOB", "NPHI/PHIN", "VCL/VCLAY/VSH", "PHIE/PHIT"],
                [rhob, nphi, vcl, phie],
            )
            if data is None
        ]
        if missing:
            raise ValueError(f"Missing required curves: {missing}")

        result = self.solve(
            rhob, nphi, vcl, phie,
            pef=pef,
            dt=dt,
        )

        curve_names = {
            "quartz":    "VQTZ",
            "calcite":   "VCAL",
            "dolomite":  "VDOL",
            "anhydrite": "VANH",
        }
        for mineral, values in result.items():
            if mineral in ("vcl", "phie", "qc_flag"):
                continue
            curve_name = curve_names.get(mineral, f"V{mineral.upper()[:4]}")
            well.add_curve(
                curve_name, values, unit="v/v",
                description=f"Volume {mineral}",
            )

        well.add_curve(
            "MMQC", result["qc_flag"].astype(float), unit="",
            description="Multi-mineral closure QC flag (1 = poor fit)",
        )

        return well


def solve_two_mineral(
    rhob: Union[float, np.ndarray],
    nphi: Union[float, np.ndarray],
    vcl: Optional[Union[float, np.ndarray]] = None,
    phie: Optional[Union[float, np.ndarray]] = None,
    minerals: Sequence[str] = ("quartz", "calcite"),
    pef: Optional[Union[float, np.ndarray]] = None,
    dt: Optional[Union[float, np.ndarray]] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Backward-compatible convenience wrapper for a 2-mineral solve.

    Older public imports exposed ``solve_two_mineral`` directly. The current
    implementation routes through ``MultiMineralSolver`` and defaults VCL/PHIE
    to zero when they are not provided.
    """
    rhob_arr = np.atleast_1d(np.asarray(rhob, dtype=float))
    nphi_arr = np.atleast_1d(np.asarray(nphi, dtype=float))
    vcl_arr = np.zeros_like(rhob_arr) if vcl is None else np.atleast_1d(np.asarray(vcl, dtype=float))
    phie_arr = np.zeros_like(rhob_arr) if phie is None else np.atleast_1d(np.asarray(phie, dtype=float))

    solver = MultiMineralSolver(
        minerals=list(minerals),
        minerals_no_pef=list(minerals),
        **kwargs,
    )
    return solver.solve(rhob_arr, nphi_arr, vcl_arr, phie_arr, pef=pef, dt=dt)


def solve_three_mineral(
    rhob: Union[float, np.ndarray],
    nphi: Union[float, np.ndarray],
    vcl: Optional[Union[float, np.ndarray]] = None,
    phie: Optional[Union[float, np.ndarray]] = None,
    minerals: Sequence[str] = ("quartz", "calcite", "dolomite"),
    pef: Optional[Union[float, np.ndarray]] = None,
    dt: Optional[Union[float, np.ndarray]] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Backward-compatible convenience wrapper for a 3-mineral solve.

    Older modules import this symbol directly; keeping it here avoids import
    failures while using the newer ``MultiMineralSolver`` implementation.
    """
    rhob_arr = np.atleast_1d(np.asarray(rhob, dtype=float))
    nphi_arr = np.atleast_1d(np.asarray(nphi, dtype=float))
    vcl_arr = np.zeros_like(rhob_arr) if vcl is None else np.atleast_1d(np.asarray(vcl, dtype=float))
    phie_arr = np.zeros_like(rhob_arr) if phie is None else np.atleast_1d(np.asarray(phie, dtype=float))

    solver = MultiMineralSolver(
        minerals=list(minerals),
        minerals_no_pef=list(minerals),
        **kwargs,
    )
    return solver.solve(rhob_arr, nphi_arr, vcl_arr, phie_arr, pef=pef, dt=dt)
