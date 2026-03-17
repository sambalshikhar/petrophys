"""
Multi-mineral solver for lithology determination.

This module provides solvers for determining mineral fractions
from multiple log measurements using simultaneous equations.
"""

from typing import Optional, Tuple, Dict, Union
import numpy as np
from scipy.optimize import nnls

from petrophys.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        "u":       _u(p["pef"], p["density"]),
        "dt":      p["dt"],
    }


# ---------------------------------------------------------------------------
# Two-mineral solver
# ---------------------------------------------------------------------------

def solve_two_mineral(
    log_value: float,
    mineral1_value: float,
    mineral2_value: float,
    fluid_value: float,
    porosity: float,
) -> Tuple[float, float]:
    """
    Solve for two mineral fractions given a log response.

    Uses the equation:
        Log = φ * Fluid + (1-φ) * (V1 * M1 + V2 * M2)
    where V1 + V2 = 1

    Parameters
    ----------
    log_value : float
        Measured log value
    mineral1_value : float
        Response of mineral 1
    mineral2_value : float
        Response of mineral 2
    fluid_value : float
        Response of fluid
    porosity : float
        Porosity (0-1)

    Returns
    -------
    tuple
        (V1, V2) mineral fractions
    """
    matrix_value = (log_value - porosity * fluid_value) / (1 - porosity)

    if mineral1_value == mineral2_value:
        v1 = 0.5
    else:
        v1 = (matrix_value - mineral2_value) / (mineral1_value - mineral2_value)

    v1 = np.clip(v1, 0, 1)
    return v1, 1 - v1


# ---------------------------------------------------------------------------
# Three-mineral solver
# ---------------------------------------------------------------------------

def solve_three_mineral(
    rhob: float,
    nphi: float,
    vcl: float,
    porosity: float,
    pef: Optional[float] = None,
    dt: Optional[float] = None,
    minerals: Tuple[str, str, str] = ("quartz", "calcite", "dolomite"),
    clay_mineral: str = "clay",
    fluid: str = "water",
) -> Tuple[float, float, float]:
    """
    Solve for three mineral fractions using RHOB, NPHI, and optionally PEF and DTC.

    Vcl and porosity are pre-computed and subtracted from the RHS of every
    equation before solving.  For each row:

        rhs = log_val - Vcl · endpoint_cl - φ · endpoint_fl

    This gives an overdetermined system (3 to 5 equations, 3 unknowns)
    solved by non-negative least squares (NNLS).

    PEF is converted to U = PEF × ρ before entering the matrix, as U
    is the quantity that mixes linearly by volume.

    Parameters
    ----------
    rhob : float
        Bulk density (g/cm³)
    nphi : float
        Neutron porosity (v/v)
    vcl : float
        Pre-computed clay volume fraction (0-1)
    porosity : float
        Pre-computed porosity (0-1)
    pef : float or None
        Photoelectric factor (b/e) — None if curve absent
    dt : float or None
        Sonic slowness (μs/ft) — None if curve absent
    minerals : tuple
        Three non-clay mineral types to solve for
    clay_mineral : str
        Clay mineral name for endpoint lookup
    fluid : str
        Fluid type

    Returns
    -------
    tuple
        (V1, V2, V3) mineral volume fractions
    """
    vcl      = float(np.clip(vcl,      0.0, 1.0))
    porosity = float(np.clip(porosity, 0.0, 1.0))

    m  = [_get_endpoints(name) for name in minerals]
    fl = _get_endpoints(fluid)
    cl = _get_endpoints(clay_mineral)

    A_rows, b_rows = [], []

    def add_row(mineral_vals, cl_val, fl_val, log_val):
        rhs = log_val - vcl * cl_val - porosity * fl_val
        A_rows.append(mineral_vals)
        b_rows.append(rhs)

    # Unity  (cl and fl endpoints are both 1)
    add_row([1.0, 1.0, 1.0],
            cl_val=1.0,        fl_val=1.0,        log_val=1.0)

    # Density
    add_row([m[0]["density"], m[1]["density"], m[2]["density"]],
            cl_val=cl["density"], fl_val=fl["density"], log_val=rhob)

    # Neutron
    add_row([m[0]["nphi"], m[1]["nphi"], m[2]["nphi"]],
            cl_val=cl["nphi"],    fl_val=fl["nphi"],    log_val=nphi)

    # U = PEF × ρ
    if pef is not None:
        u_log = _u(pef, rhob)
        add_row([m[0]["u"], m[1]["u"], m[2]["u"]],
                cl_val=cl["u"],   fl_val=fl["u"],   log_val=u_log)

    # Sonic (DTC)
    if dt is not None:
        add_row([m[0]["dt"], m[1]["dt"], m[2]["dt"]],
                cl_val=cl["dt"],  fl_val=fl["dt"],  log_val=dt)

    x, _ = nnls(np.array(A_rows), np.array(b_rows))

    # Check closure against available mineral space (1 - Vcl - φ)
    target = 1.0 - vcl - porosity
    if abs(x.sum() - target) > 0.05:
        return np.nan, np.nan, np.nan

    return float(x[0]), float(x[1]), float(x[2])


# ---------------------------------------------------------------------------
# MultiMineralSolver class
# ---------------------------------------------------------------------------

class MultiMineralSolver:
    """
    Multi-mineral solver for complex lithology.

    This class solves for mineral fractions using multiple
    log measurements simultaneously.

    Vcl (from VclModule) and porosity (from PorosityModule) must be
    pre-computed before calling this solver.  The matrix is built
    dynamically from available curves — PEF and DTC are optional
    and each adds one equation when present.

    Parameters
    ----------
    minerals : list, optional
        List of three non-clay mineral types to solve for
    clay_mineral : str, optional
        Clay mineral name for endpoint lookup (default: "clay")
    fluid : str, optional
        Fluid type (default: "water")
    use_pef : bool, optional
        Include PEF (as U = PEF × ρ) in solution (default: True)
    use_dt : bool, optional
        Include DTC in solution (default: True)

    Examples
    --------
    >>> solver = MultiMineralSolver(minerals=["quartz", "calcite", "dolomite"])
    >>> result = solver.solve(rhob, nphi, vcl, porosity, pef=pef, dt=dt)
    """

    def __init__(
        self,
        minerals: list = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
        use_pef: bool = True,
        use_dt: bool = True,
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite"]

        self.minerals     = minerals
        self.clay_mineral = clay_mineral
        self.fluid        = fluid
        self.use_pef      = use_pef
        self.use_dt       = use_dt

        self._load_properties()

    def _load_properties(self):
        """Load mineral and fluid properties."""
        self.mineral_props = {m: _get_endpoints(m) for m in self.minerals}
        self.fluid_props   = _get_endpoints(self.fluid)
        self.clay_props    = _get_endpoints(self.clay_mineral)

    def solve(
        self,
        rhob:     Union[float, np.ndarray],
        nphi:     Union[float, np.ndarray],
        vcl:      Union[float, np.ndarray],
        porosity: Union[float, np.ndarray],
        pef:      Optional[Union[float, np.ndarray]] = None,
        dt:       Optional[Union[float, np.ndarray]] = None,
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
        porosity : float or np.ndarray
            Pre-computed porosity (0-1)
        pef : float or np.ndarray, optional
            Photoelectric factor (b/e) — omit if curve absent
        dt : float or np.ndarray, optional
            Sonic slowness (μs/ft) — omit if curve absent

        Returns
        -------
        dict
            Dictionary with mineral fractions, vcl, porosity, and qc_flag
        """
        rhob     = np.atleast_1d(np.asarray(rhob,     dtype=float))
        nphi     = np.atleast_1d(np.asarray(nphi,     dtype=float))
        vcl      = np.atleast_1d(np.asarray(vcl,      dtype=float))
        porosity = np.atleast_1d(np.asarray(porosity, dtype=float))
        n        = len(rhob)

        pef_arr = np.atleast_1d(np.asarray(pef, dtype=float)) if pef is not None else None
        dt_arr  = np.atleast_1d(np.asarray(dt,  dtype=float)) if dt  is not None else None

        result = {mineral: np.zeros(n) for mineral in self.minerals}
        result["vcl"]      = vcl.copy()
        result["porosity"] = porosity.copy()
        result["qc_flag"]  = np.zeros(n, dtype=bool)

        for i in range(n):
            if any(np.isnan([rhob[i], nphi[i], vcl[i], porosity[i]])):
                for mineral in self.minerals:
                    result[mineral][i] = np.nan
                result["qc_flag"][i] = True
                continue

            # Per-depth: treat NaN optional samples as absent
            pef_i = float(pef_arr[i]) if pef_arr is not None else None
            dt_i  = float(dt_arr[i])  if dt_arr  is not None else None

            if pef_i is not None and (np.isnan(pef_i) or not self.use_pef):
                pef_i = None
            if dt_i  is not None and (np.isnan(dt_i)  or not self.use_dt):
                dt_i  = None

            if len(self.minerals) == 3:
                v1, v2, v3 = solve_three_mineral(
                    rhob[i], nphi[i], vcl[i], porosity[i],
                    pef=pef_i, dt=dt_i,
                    minerals=tuple(self.minerals),
                    clay_mineral=self.clay_mineral,
                    fluid=self.fluid,
                )
                result[self.minerals[0]][i] = v1
                result[self.minerals[1]][i] = v2
                result[self.minerals[2]][i] = v3

                if any(np.isnan([v1, v2, v3])):
                    result["qc_flag"][i] = True

            elif len(self.minerals) == 2:
                v1, v2 = solve_two_mineral(
                    rhob[i],
                    self.mineral_props[self.minerals[0]]["density"],
                    self.mineral_props[self.minerals[1]]["density"],
                    self.fluid_props["density"],
                    porosity[i],
                )
                result[self.minerals[0]][i] = v1
                result[self.minerals[1]][i] = v2

            else:
                for mineral in self.minerals:
                    result[mineral][i] = np.nan
                result["qc_flag"][i] = True

        return result

    def add_to_well(self, well: "Well") -> "Well":
        """
        Add lithology curves to well.

        Parameters
        ----------
        well : Well
            Well object. VCL must have been added by VclModule,
            PHIE by PorosityModule, before calling this.

        Returns
        -------
        Well
            Well with lithology curves added
        """
        rhob = well.get("RHOB")
        nphi = well.get("NPHI")
        vcl  = well.get("VCL")
        phie = well.get("PHIE")
        pef  = well.get("PEF")
        dt   = well.get("DTC")

        missing = [
            name for name, c in zip(
                ["RHOB", "NPHI", "VCL", "PHIE"],
                [rhob, nphi, vcl, phie]
            ) if c is None
        ]
        if missing:
            raise ValueError(f"Missing required curves: {missing}")

        result = self.solve(
            rhob.data, nphi.data, vcl.data, phie.data,
            pef=pef.data if pef is not None else None,
            dt=dt.data   if dt  is not None else None,
        )

        curve_names = {
            "quartz":    "VQTZ",
            "calcite":   "VCAL",
            "dolomite":  "VDOL",
            "anhydrite": "VANH",
        }
        for mineral, values in result.items():
            if mineral in ("vcl", "porosity", "qc_flag"):
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
