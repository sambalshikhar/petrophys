"""
Multi-mineral solver, version 1.

Version 1 is the original solver, used by the dynamic workflow's clastic
mineral step. VCL and PHIE are taken as fixed inputs
and moved to the right-hand side of the tool-response equations, the mineral
volumes are solved by non-negative least squares, and an absolute closure gate
rejects samples whose solved mineral total misses the available rock volume
``1 - Vcl - PHIE`` by more than 0.05. Rejected samples are left NaN and
flagged, so mineral volumes are only reported where the logs, VCL, and PHIE
agree.

The equations carry different units (v/v, g/cm³, b/cm³, μs/ft), so each row is
divided by its nominal one-sigma uncertainty (:data:`ROW_UNCERTAINTY`) before
the solve: every residual is then measured in units of tool error, the
weighted least squares of Mayer and Sibbit (1980, SPE 9341) and Quirein et al.
(1986, SPWLA 27th Symposium). Solved unscaled (``row_uncertainty=None``), the
sonic and U rows dominate the fit and the closure gate rejects most samples
whose logs agree to within tool precision.

Version 2 (:class:`~petrophys.interpretation.lithology.MultiMineralSolver`)
solves porosity jointly with the minerals (or constrains it with core
porosity), so the mineral result does not depend on a pre-computed PHIE, and
reports underdetermined samples instead of gating them.
"""

from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.optimize import nnls

from petrophys.interpretation.lithology.matrix_properties import get_log_density, get_mineral_properties


# Tool-response endpoints frozen with version 1, so its results cannot drift
# when the shared mineral table changes. Other minerals use the table.
_V1_ENDPOINTS: Dict[str, Dict[str, float]] = {
    "quartz":    {"density": 2.65, "nphi": -0.02, "pef": 1.8,  "u": 4.8,  "dt": 55.5},
    "calcite":   {"density": 2.71, "nphi": 0.0,   "pef": 5.1,  "u": 13.8, "dt": 47.6},
    "dolomite":  {"density": 2.87, "nphi": 0.02,  "pef": 3.1,  "u": 9.0,  "dt": 43.5},
    "anhydrite": {"density": 2.96, "nphi": -0.02, "pef": 5.0,  "u": 14.9, "dt": 50.0},
    "clay":      {"density": 2.65, "nphi": 0.35,  "pef": 3.7,  "u": 9.71, "dt": 80.0},
    "water":     {"density": 1.0,  "nphi": 1.0,   "pef": 0.36, "u": 0.36, "dt": 189.0},
}

CLOSURE_TOLERANCE = 0.05

# Nominal one-sigma uncertainty of each equation row, in the row's own unit.
# The unity (closure) row is weighted most strongly: the 0.05 closure gate is
# five of its sigmas. Sonic follows pore shape and compaction as much as
# mineralogy: beside PEF it helps the closure, but it does not replace PEF
# (see ``sonic_without_pef``).
ROW_UNCERTAINTY: Mapping[str, float] = MappingProxyType({
    "unity": 0.01,  # v/v
    "rhob":  0.02,  # g/cm³
    "nphi":  0.02,  # v/v
    "u":     0.5,   # b/cm³
    "dt":    2.0,   # μs/ft
})


def _get_endpoints(name: str) -> Dict[str, float]:
    """Return endpoint dict with density, nphi, pef, u, dt for a mineral/fluid."""
    if name in _V1_ENDPOINTS:
        return dict(_V1_ENDPOINTS[name])
    p = get_mineral_properties(name)
    return {
        # the density-log response (evaporites read below their grain density)
        "density": get_log_density(name),
        "nphi":    p["nphi"],
        "pef":     p["pef"],
        "u":       p["u"],
        "dt":      p["slowness"],
    }


def _row_uncertainty(
    row_uncertainty: Optional[Mapping[str, float]],
) -> Optional[Dict[str, float]]:
    """Merge user row uncertainties over :data:`ROW_UNCERTAINTY`; None stays None."""
    if row_uncertainty is None:
        return None
    unknown = sorted(set(row_uncertainty) - set(ROW_UNCERTAINTY))
    if unknown:
        raise ValueError(
            f"row_uncertainty keys must be among {', '.join(ROW_UNCERTAINTY)}; got {', '.join(unknown)}"
        )
    merged = {**ROW_UNCERTAINTY, **row_uncertainty}
    for row, sigma in merged.items():
        number = isinstance(sigma, (int, float, np.integer, np.floating)) and not isinstance(sigma, bool)
        if not (number and np.isfinite(sigma) and sigma > 0):
            raise ValueError(f"row_uncertainty[{row!r}] must be a positive number; got {sigma!r}")
    return {row: float(sigma) for row, sigma in merged.items()}


def _constrain_mineral_fractions(
    fractions: np.ndarray,
    available_fraction: float,
) -> np.ndarray:
    """Clip mineral fractions and scale them down if they exceed the available rock volume."""
    constrained = np.asarray(fractions, dtype=float).copy()
    constrained = np.clip(constrained, 0.0, max(float(available_fraction), 0.0))
    total = float(np.nansum(constrained))
    if total > available_fraction and total > 0.0:
        constrained *= float(available_fraction) / total
    return constrained


class MultiMineralSolverV1:
    """
    Multi-mineral solver, version 1: fixed VCL and PHIE with a closure gate.

    VCL (from VclModule) and PHIE (from PorosityModule) must be pre-computed
    before calling this solver. Both are subtracted from the right-hand side of
    every equation before solving for the mineral volumes.

    The active mineral set and equations are determined per depth from curve
    availability::

        PEF available              -> 4 minerals (quartz, calcite, dolomite,
                                      anhydrite) + PEF row (+ DTC if present)
        PEF absent                 -> 3 minerals, RHOB + NPHI only; with
                                      ``sonic_without_pef`` a DTC row makes it
                                      4 minerals + DTC row

    Tool Response Equations (linear mixing)::

        Unity:   V1 + V2 + ... + Vn         = 1    - Vcl - φe
        Density: V1·ρ1 + V2·ρ2 + ... + Vn·ρn = ρlog - Vcl·ρcl - φe·ρfl
        Neutron: V1·N1 + V2·N2 + ... + Vn·Nn = Nlog - Vcl·Ncl - φe·Nfl
        U:       V1·U1 + V2·U2 + ... + Vn·Un = Ulog - Vcl·Ucl - φe·Ufl
        Sonic:   V1·D1 + V2·D2 + ... + Vn·Dn = Dlog - Vcl·Dcl - φe·Dfl

    where U = PEF × ρ. Each row is divided by its ``row_uncertainty`` before
    the non-negative least-squares solve. When the solved total differs from
    ``1 - Vcl - φe`` by more than 0.05 the sample is not solved: its mineral
    volumes are NaN and ``qc_flag`` is set. Otherwise volumes are clipped to
    the available rock volume and scaled down only when their total exceeds
    it; a total below it (within the 0.05 gate) is reported unscaled.

    Parameters
    ----------
    minerals : list, optional
        Minerals solved when PEF or DTC is available
        (default: ["quartz", "calcite", "dolomite", "anhydrite"]).
    minerals_no_pef : list, optional
        Minerals solved when neither PEF nor DTC is available
        (default: the first three of ``minerals``).
    clay_mineral : str, optional
        Clay endpoint name (default: "clay").
    fluid : str, optional
        Pore-fluid endpoint name (default: "water").
    row_uncertainty : mapping or None, optional
        One-sigma uncertainty of each equation row, keyed ``unity`` (v/v),
        ``rhob`` (g/cm³), ``nphi`` (v/v), ``u`` (b/cm³) and ``dt`` (μs/ft).
        Rows are divided by these values before the solve; keys left out keep
        :data:`ROW_UNCERTAINTY` (0.01, 0.02, 0.02, 0.5, 2.0). None solves the
        rows unscaled, the calculation behind the "geolink" calculation profile.
    sonic_without_pef : bool, optional
        Let a sonic row stand in for a missing PEF and solve the four-mineral
        set (the original behaviour, kept by the benchmark profiles). Default
        False: without PEF the three-mineral set is solved from density and
        neutron. Against vendor lithology logs (15/9-14, 30/6-5) the sonic
        stand-in called 84-97 % of no-PEF carbonates quartz; density and
        neutron alone called 74-89 % of them carbonate.
    endpoint_overrides : mapping, optional
        Endpoint values replacing the table's, keyed by endpoint name and then
        ``density`` (g/cm³), ``nphi``, ``pef``, ``u`` or ``dt`` (μs/ft): e.g.
        ``{"water": {"density": 1.1}, "clay": {"density": 2.45}}``. The
        ``fluid`` and ``clay_mineral`` names address the fluid and clay.

    Examples
    --------
    >>> solver = MultiMineralSolverV1()
    >>> result = solver.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)
    """

    version = 1

    def __init__(
        self,
        minerals: Optional[list] = None,
        minerals_no_pef: Optional[list] = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
        row_uncertainty: Optional[Mapping[str, float]] = ROW_UNCERTAINTY,
        sonic_without_pef: bool = False,
        endpoint_overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]
        if minerals_no_pef is None:
            minerals_no_pef = minerals[:3]

        self.minerals        = minerals
        self.minerals_no_pef = minerals_no_pef
        self.clay_mineral    = clay_mineral
        self.fluid           = fluid
        self.row_uncertainty = _row_uncertainty(row_uncertainty)
        self.sonic_without_pef = bool(sonic_without_pef)
        self.endpoint_overrides = {name: dict(values) for name, values in (endpoint_overrides or {}).items()}

        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))
        self.mineral_props = {m: _get_endpoints(m) for m in all_minerals}
        self.fluid_props   = _get_endpoints(self.fluid)
        self.clay_props    = _get_endpoints(self.clay_mineral)
        for name, values in (self.endpoint_overrides or {}).items():
            if name == self.fluid:
                self.fluid_props.update(values)
            if name == self.clay_mineral:
                self.clay_props.update(values)
            if name in self.mineral_props:
                self.mineral_props[name].update(values)

    def _solve_sample(
        self,
        rhob: float,
        nphi: float,
        vcl: float,
        phie: float,
        pef: Optional[float],
        dt: Optional[float],
    ) -> Tuple[list, Tuple[float, ...]]:
        """Return the active minerals and their volumes (NaN when the closure gate fails)."""
        vcl  = float(np.clip(vcl,  0.0, 1.0))
        phie = float(np.clip(phie, 0.0, 1.0))

        # 4 minerals need at least 4 equations: Unity + RHOB + NPHI + (PEF or DTC).
        if pef is None and not self.sonic_without_pef:
            dt = None
        has_discriminator = pef is not None or dt is not None
        active_minerals = self.minerals if has_discriminator else self.minerals_no_pef

        m  = [self.mineral_props[name] for name in active_minerals]
        fl = self.fluid_props
        cl = self.clay_props

        A_rows, b_rows, row_names = [], [], []

        def add_row(row, mineral_vals, cl_val, fl_val, log_val):
            A_rows.append(mineral_vals)
            b_rows.append(log_val - vcl * cl_val - phie * fl_val)
            row_names.append(row)

        add_row("unity", [1.0] * len(active_minerals), cl_val=1.0, fl_val=1.0, log_val=1.0)
        add_row("rhob", [mi["density"] for mi in m], cl_val=cl["density"], fl_val=fl["density"], log_val=rhob)
        add_row("nphi", [mi["nphi"] for mi in m], cl_val=cl["nphi"], fl_val=fl["nphi"], log_val=nphi)
        if pef is not None:
            add_row("u", [mi["u"] for mi in m], cl_val=cl["u"], fl_val=fl["u"], log_val=pef * rhob)
        if dt is not None:
            add_row("dt", [mi["dt"] for mi in m], cl_val=cl["dt"], fl_val=fl["dt"], log_val=dt)

        A, b = np.array(A_rows), np.array(b_rows)
        if self.row_uncertainty is not None:
            weights = 1.0 / np.array([self.row_uncertainty[row] for row in row_names])
            A, b = A * weights[:, None], b * weights
        x, _ = nnls(A, b)

        available_fraction = 1.0 - vcl - phie
        if abs(x.sum() - available_fraction) > CLOSURE_TOLERANCE:
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
        core_phi: Optional[Union[float, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve for mineral fractions.

        Parameters
        ----------
        rhob, nphi : float or np.ndarray
            Bulk density (g/cm³) and neutron porosity (v/v).
        vcl, phie : float or np.ndarray
            Pre-computed clay volume and effective porosity (0-1).
        pef, dt : float or np.ndarray, optional
            Photoelectric factor (b/e) and sonic slowness (μs/ft).
        core_phi : optional
            Accepted for interface compatibility with version 2 and ignored;
            version 1 always uses ``phie``.

        Returns
        -------
        dict
            Volumes for every mineral in ``minerals`` and ``minerals_no_pef``
            (0 where a mineral was not active at that depth, NaN where the
            sample was not solved), the input ``vcl`` and ``phie``,
            ``qc_flag`` (True where not solved) and ``solved`` (its inverse).
        """
        del core_phi
        rhob = np.atleast_1d(np.asarray(rhob, dtype=float))
        nphi = np.atleast_1d(np.asarray(nphi, dtype=float))
        vcl  = np.atleast_1d(np.asarray(vcl,  dtype=float))
        phie = np.atleast_1d(np.asarray(phie, dtype=float))
        n    = len(rhob)

        pef_arr = np.atleast_1d(np.asarray(pef, dtype=float)) if pef is not None else None
        dt_arr  = np.atleast_1d(np.asarray(dt,  dtype=float)) if dt  is not None else None

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
            if dt_i is not None and np.isnan(dt_i):
                dt_i = None

            active_minerals, fractions = self._solve_sample(rhob[i], nphi[i], vcl[i], phie[i], pef_i, dt_i)
            for mineral, v in zip(active_minerals, fractions):
                result[mineral][i] = v
            if any(np.isnan(fractions)):
                result["qc_flag"][i] = True

        result["solved"] = ~result["qc_flag"]
        return result
