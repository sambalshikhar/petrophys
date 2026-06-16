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

    The active mineral set and equations are determined per depth from the
    equations available at that sample. Unity and RHOB are always used; NPHI,
    PEF, and DTC are optional discriminator rows. The full mineral set is used
    when enough independent rows are available, otherwise the solver falls
    back to the reduced mineral set.

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
    >>> result = solver.solve(rhob, nphi=None, vcl=vcl, phie=phie, pef=pef, dt=dt)
    """

    def __init__(
        self,
        minerals: list = None,
        minerals_no_pef: list = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
        closure_tolerance: float = 0.05,
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]
        if minerals_no_pef is None:
            minerals_no_pef = minerals[:3]

        self.minerals          = minerals
        self.minerals_no_pef   = minerals_no_pef
        self.clay_mineral      = clay_mineral
        self.fluid             = fluid
        self.closure_tolerance = closure_tolerance

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
        nphi: Optional[float],
        vcl: float,
        phie: float,
        pef: Optional[float],
        dt: Optional[float],
    ) -> Tuple[list, tuple, float, bool]:
        """
        Solve for mineral fractions at a single depth sample.

        Active mineral set is chosen from equation count. Unity and RHOB are
        always present. NPHI, PEF, and DTC are optional rows; neutron is not
        required when PEF/DTC provide enough independent equations.

        Parameters
        ----------
        rhob : float
            Bulk density (g/cm³)
        nphi : float or None
            Neutron porosity (v/v), omitted when unavailable.
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
            (active_minerals, fractions, residual, poor_fit) where
            active_minerals is the list of mineral names used,
            fractions are the solved (normalized) volumes,
            residual is the NNLS least-squares residual, and
            poor_fit is True when closure error exceeds closure_tolerance.
        """
        vcl  = float(np.clip(vcl,  0.0, 1.0))
        phie = float(np.clip(phie, 0.0, 1.0))

        available_equations = 2 + sum(value is not None for value in (nphi, pef, dt))
        if available_equations >= len(self.minerals):
            active_minerals = self.minerals
        elif available_equations >= len(self.minerals_no_pef):
            active_minerals = self.minerals_no_pef
        else:
            return [], tuple(), np.nan, True

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
        if nphi is not None:
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

        x, residual = nnls(np.array(A_rows), np.array(b_rows))

        available_fraction = 1.0 - vcl - phie
        closure_error = abs(x.sum() - available_fraction)
        poor_fit = closure_error > self.closure_tolerance

        # Always normalize — never discard a solution just for closure error.
        # The poor_fit flag and residual give the caller full quality info.
        x = _constrain_mineral_fractions(x, available_fraction)

        return active_minerals, tuple(float(v) for v in x), float(residual), poor_fit

    def solve(
        self,
        rhob: Union[float, np.ndarray],
        nphi: Optional[Union[float, np.ndarray]],
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
        nphi : float or np.ndarray, optional
            Neutron porosity (v/v). Pass None when unavailable.
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
        raw_arrays = {
            "rhob": np.atleast_1d(np.asarray(rhob, dtype=float)),
            "vcl": np.atleast_1d(np.asarray(vcl, dtype=float)),
            "phie": np.atleast_1d(np.asarray(phie, dtype=float)),
        }
        if nphi is not None:
            raw_arrays["nphi"] = np.atleast_1d(np.asarray(nphi, dtype=float))
        if pef is not None:
            raw_arrays["pef"] = np.atleast_1d(np.asarray(pef, dtype=float))
        if dt is not None:
            raw_arrays["dt"] = np.atleast_1d(np.asarray(dt, dtype=float))

        lengths = [len(values) for values in raw_arrays.values()]
        n = max(lengths)

        def _broadcast(name: str, values: np.ndarray) -> np.ndarray:
            if len(values) == n:
                return values.astype(float, copy=False)
            if len(values) == 1:
                return np.full(n, float(values[0]), dtype=float)
            raise ValueError(
                f"{name} length ({len(values)}) must be 1 or match solve length ({n})"
            )

        rhob = _broadcast("rhob", raw_arrays["rhob"])
        nphi_arr = _broadcast("nphi", raw_arrays["nphi"]) if nphi is not None else None
        vcl = _broadcast("vcl", raw_arrays["vcl"])
        phie = _broadcast("phie", raw_arrays["phie"])
        pef_arr = _broadcast("pef", raw_arrays["pef"]) if pef is not None else None
        dt_arr = _broadcast("dt", raw_arrays["dt"]) if dt is not None else None

        # All possible minerals across both sets
        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))

        result = {mineral: np.zeros(n) for mineral in all_minerals}
        result["vcl"]      = vcl.copy()
        result["phie"]     = phie.copy()
        result["residual"] = np.zeros(n)
        result["qc_flag"]  = np.zeros(n, dtype=bool)
        result["solved"]   = np.zeros(n, dtype=bool)

        for i in range(n):
            if any(np.isnan([rhob[i], vcl[i], phie[i]])):
                for mineral in all_minerals:
                    result[mineral][i] = np.nan
                result["residual"][i] = np.nan
                result["qc_flag"][i]  = True
                continue

            nphi_i = float(nphi_arr[i]) if nphi_arr is not None else None
            pef_i = float(pef_arr[i]) if pef_arr is not None else None
            dt_i  = float(dt_arr[i])  if dt_arr  is not None else None

            if nphi_i is not None and np.isnan(nphi_i):
                nphi_i = None
            if pef_i is not None and np.isnan(pef_i):
                pef_i = None
            if dt_i  is not None and np.isnan(dt_i):
                dt_i  = None

            active_minerals, fractions, residual, poor_fit = self._solve_sample(
                rhob[i], nphi_i, vcl[i], phie[i], pef_i, dt_i
            )

            if not active_minerals:
                for mineral in all_minerals:
                    result[mineral][i] = np.nan
                result["residual"][i] = np.nan
                result["qc_flag"][i] = True
                continue

            for mineral, v in zip(active_minerals, fractions):
                result[mineral][i] = v

            result["residual"][i] = residual
            result["qc_flag"][i]  = poor_fit
            result["solved"][i] = True

        return result

    def add_to_well(self, well: "Well") -> "Well":
        """
        Add lithology curves to well.

        Requires RHOB plus VCL and PHIE/PHIT on the well object. At least one
        discriminator curve from NPHI/PHIN, PEF, or DTC/DT must be present.
        Neutron is helpful but not mandatory when PEF/DTC provide enough
        independent equations.

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
                ["RHOB", "VCL/VCLAY/VSH", "PHIE/PHIT"],
                [rhob, vcl, phie],
            )
            if data is None
        ]
        if nphi is None and pef is None and dt is None:
            missing.append("NPHI/PHIN, PEF, or DTC/DT")
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
            if mineral in ("vcl", "phie", "residual", "qc_flag", "solved"):
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
    nphi: Optional[Union[float, np.ndarray]] = None,
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
    nphi_arr = None if nphi is None else np.atleast_1d(np.asarray(nphi, dtype=float))
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
    nphi: Optional[Union[float, np.ndarray]] = None,
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
    nphi_arr = None if nphi is None else np.atleast_1d(np.asarray(nphi, dtype=float))
    vcl_arr = np.zeros_like(rhob_arr) if vcl is None else np.atleast_1d(np.asarray(vcl, dtype=float))
    phie_arr = np.zeros_like(rhob_arr) if phie is None else np.atleast_1d(np.asarray(phie, dtype=float))

    solver = MultiMineralSolver(
        minerals=list(minerals),
        minerals_no_pef=list(minerals),
        **kwargs,
    )
    return solver.solve(rhob_arr, nphi_arr, vcl_arr, phie_arr, pef=pef, dt=dt)
