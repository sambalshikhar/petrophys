"""
Multi-mineral solver: mineral volumes from the tool-response equations.

Each log reads a linear volumetric mix of its response to every mineral, the
clay and the pore fluid. One equation per log, with a unity equation that
closes the volumes, is solved by non-negative least squares for the mineral
volumes. The equations carry different units (v/v, g/cm³, b/cm³, μs/ft), so
each row is divided by its nominal one-sigma tool uncertainty
(:data:`ROW_UNCERTAINTY`) first: every residual is then measured in units of
tool error, the weighted least squares of Mayer and Sibbit (1980, SPE 9341)
and Quirein et al. (1986, SPWLA 27th Symposium).

Porosity is taken from the interpretation (PHIE) where the sample has a
neutron log and a porosity; elsewhere it is solved with the minerals, from
density and the photoelectric factor, so the minerals do not rest on a
porosity that assumed a matrix.

References
----------
Mayer, C. and Sibbit, A. (1980). GLOBAL, a new approach to
computer-processed log interpretation. SPE 9341.

Quirein, J., Kimminau, S., Lavigne, J., Singer, J. and Wendel, F. (1986).
A coherent framework for developing and applying multiple formation
evaluation models. SPWLA 27th Annual Logging Symposium.

Doveton, J.H. (1994). *Geologic Log Analysis Using Computer Methods*.
AAPG Computer Applications in Geology 2.

Lawson, C.L. and Hanson, R.J. (1974). *Solving Least Squares Problems*.
Prentice-Hall.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys._internal.nnls import nnls, nnls_many
from petrophys.interpretation.lithology.matrix_properties import get_log_density, mineral_properties

if TYPE_CHECKING:
    from petrophys.model.well import Well


# Tool-response endpoints of the solver, held apart from the shared mineral
# table so its results do not follow that table. Other minerals use the table.
_ENDPOINTS: Dict[str, Dict[str, float]] = {
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

_ROW_LABELS = {"unity": "Unity", "rhob": "Density (RHOB)", "nphi": "Neutron (NPHI)",
               "u": "U = PEF·ρ", "dt": "Sonic (DT)"}


def _get_endpoints(name: str) -> Dict[str, float]:
    """Return endpoint dict with density, nphi, pef, u, dt for a mineral/fluid."""
    if name in _ENDPOINTS:
        return dict(_ENDPOINTS[name])
    p = mineral_properties(name)
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


def _finite(value: Optional[float]) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else float(value)


class MultiMineralSolver:
    """
    Multi-mineral solver: quartz, calcite, dolomite and anhydrite volumes.

    VCL (from VclModule) is a fixed input. Porosity is fixed or solved per
    sample:

    * **PHIE given** where the sample has NPHI and a finite ``phie``: VCL
      and PHIE are subtracted from the right-hand side and the minerals
      solved. When the solved total misses ``1 - VCL - PHIE`` by more than
      0.05, the sample is not solved (NaN volumes, ``qc_flag`` set).
    * **Porosity solved** elsewhere (no NPHI, or no PHIE as on a raw well):
      porosity is one more unknown, so the minerals do not rest on a
      porosity computed with an assumed matrix. A core porosity
      (``core_phi``) fixes it where one is given. Density and closure with
      PEF or with NPHI give porosity and two minerals, the
      density-photoelectric or density-neutron lithology of the chartbooks;
      sonic is fitted beside them but does not stand in for either. More
      rows (NPHI and PEF together) solve more minerals, up to ``minerals``; with fewer
      the minerals come from ``minerals_reduced`` in its order (quartz and
      calcite by default, calcite and dolomite in carbonate). The same 0.05
      closure check applies.

    The active mineral set where PHIE is given::

        PEF available              -> ``minerals`` (quartz, calcite, dolomite,
                                      anhydrite) + PEF row (+ DTC if present)
        PEF absent                 -> ``minerals_no_pef``, RHOB + NPHI only;
                                      with ``sonic_without_pef`` a DTC row
                                      makes it ``minerals`` + DTC row

    Tool Response Equations (linear mixing), with porosity φ::

        Unity:   V1 + ... + Vn + φ          = 1    - Vcl
        Density: V1·ρ1 + ... + Vn·ρn + φ·ρfl = ρlog - Vcl·ρcl
        Neutron: V1·N1 + ... + Vn·Nn + φ·Nfl = Nlog - Vcl·Ncl
        U:       V1·U1 + ... + Vn·Un + φ·Ufl = Ulog - Vcl·Ucl
        Sonic:   V1·D1 + ... + Vn·Dn + φ·Dfl = Dlog - Vcl·Dcl

    where U = PEF × ρ; a fixed φ moves to the right-hand side. Each row is
    divided by its ``row_uncertainty`` before the non-negative least-squares
    solve. Solved volumes are clipped to the available rock volume and scaled
    down only when their total exceeds it.

    Parameters
    ----------
    minerals : list, optional
        Full mineral set (default: ["quartz", "calcite", "dolomite",
        "anhydrite"]).
    minerals_no_pef : list, optional
        Minerals solved where PHIE is given and PEF is not (default: the
        first three of ``minerals``).
    minerals_reduced : list, optional
        Minerals, in order of preference, solved where porosity is solved and
        the logs cannot resolve ``minerals`` (default: ["quartz",
        "calcite"]).
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
        Let a sonic row stand in for a missing PEF and solve ``minerals``
        (the calculation of the benchmark profiles). Default False: sonic is
        used beside PEF, not in its place, as it follows pore shape and
        compaction as much as mineralogy.
    solve_porosity : bool, optional
        Solve porosity with the minerals where NPHI or PHIE is missing
        (default True). False leaves those samples unsolved, the calculation
        of the benchmark profiles.
    endpoint_overrides : mapping, optional
        Endpoint values replacing the table's, keyed by endpoint name and then
        ``density`` (g/cm³), ``nphi``, ``pef``, ``u`` or ``dt`` (μs/ft): e.g.
        ``{"water": {"density": 1.1}, "clay": {"density": 2.45}}``. The
        ``fluid`` and ``clay_mineral`` names address the fluid and clay.

    Examples
    --------
    >>> solver = MultiMineralSolver()
    >>> result = solver.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)  # doctest: +SKIP
    """

    def __init__(
        self,
        minerals: Optional[list] = None,
        minerals_no_pef: Optional[list] = None,
        minerals_reduced: Optional[list] = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
        row_uncertainty: Optional[Mapping[str, float]] = ROW_UNCERTAINTY,
        sonic_without_pef: bool = False,
        solve_porosity: bool = True,
        endpoint_overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]
        if minerals_no_pef is None:
            minerals_no_pef = minerals[:3]
        if minerals_reduced is None:
            minerals_reduced = [name for name in ("quartz", "calcite") if name in minerals] or minerals[:2]

        self.minerals         = minerals
        self.minerals_no_pef  = minerals_no_pef
        self.minerals_reduced = minerals_reduced
        self.clay_mineral     = clay_mineral
        self.fluid            = fluid
        self.row_uncertainty  = _row_uncertainty(row_uncertainty)
        self.sonic_without_pef = bool(sonic_without_pef)
        self.solve_porosity   = bool(solve_porosity)
        self.endpoint_overrides = {name: dict(values) for name, values in (endpoint_overrides or {}).items()}

        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef + self.minerals_reduced))
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

    # -- the system of one sample ----------------------------------------------

    def _rows(self, nphi, pef, dt) -> List[str]:
        """Equation rows a sample's logs give; sonic only beside PEF unless ``sonic_without_pef``."""
        rows = ["unity", "rhob"]
        if nphi is not None:
            rows.append("nphi")
        if pef is not None:
            rows.append("u")
        if dt is not None and (pef is not None or self.sonic_without_pef):
            rows.append("dt")
        return rows

    def _fixed_minerals(self, pef, dt) -> list:
        """Minerals solved where PHIE is given."""
        if pef is None and not self.sonic_without_pef:
            dt = None
        return self.minerals if (pef is not None or dt is not None) else self.minerals_no_pef

    def _solved_minerals(self, rows: Sequence[str], porosity_fixed: bool) -> list:
        """Minerals the rows resolve where porosity is solved (fixed by core): [] when too few.

        Minerals beyond what the rows resolve are dropped from the end of
        ``minerals_reduced``. Sonic is fitted but not counted: it sits beside
        PEF and does not stand in for PEF or the neutron, so RHOB with sonic
        alone resolves no mineral.
        """
        resolvable = len([row for row in rows if row != "dt"]) - (0 if porosity_fixed else 1)
        if resolvable >= len(self.minerals):
            return list(self.minerals)
        reduced = list(self.minerals_reduced)[:resolvable]
        return reduced if len(reduced) >= 2 else []

    def _system(self, minerals, rows, rhob, nphi, vcl, pef, dt, porosity):
        """Response matrix and right-hand side; ``porosity`` None adds it as an unknown."""
        m = [self.mineral_props[name] for name in minerals]
        fl, cl = self.fluid_props, self.clay_props
        key = {"unity": None, "rhob": "density", "nphi": "nphi", "u": "u", "dt": "dt"}
        reading = {"unity": 1.0, "rhob": rhob, "nphi": nphi,
                   "u": None if pef is None else pef * rhob, "dt": dt}
        A_rows, b_rows = [], []
        for row in rows:
            k = key[row]
            mineral_vals = [1.0] * len(m) if k is None else [mi[k] for mi in m]
            fl_val = 1.0 if k is None else fl[k]
            cl_val = 1.0 if k is None else cl[k]
            if porosity is None:
                A_rows.append(mineral_vals + [fl_val])
                b_rows.append(reading[row] - vcl * cl_val)
            else:
                A_rows.append(mineral_vals)
                b_rows.append(reading[row] - vcl * cl_val - porosity * fl_val)
        return np.array(A_rows, dtype=float), np.array(b_rows, dtype=float)

    def _weighted(self, A, B, rows):
        if self.row_uncertainty is None:
            return A, B
        weights = 1.0 / np.array([self.row_uncertainty[row] for row in rows])
        return A * weights[:, None], B * (weights if B.ndim == 1 else weights[None, :])

    def _solve_porosity_sample(self, rhob, nphi, vcl, pef, dt, core_phi):
        """Minerals and porosity of one sample whose porosity is solved (or fixed by core).

        Returns (minerals, volumes, porosity, closed); minerals is [] when the
        logs cannot resolve two minerals.
        """
        vcl = float(np.clip(vcl, 0.0, 1.0))
        porosity = None if core_phi is None else float(np.clip(core_phi, 0.0, 1.0 - vcl))
        rows = self._rows(nphi, pef, dt)
        minerals = self._solved_minerals(rows, porosity is not None)
        if not minerals:
            return [], (), np.nan, False
        A, b = self._system(minerals, rows, rhob, nphi, vcl, pef, dt, porosity)
        if np.linalg.matrix_rank(A) < A.shape[1]:
            return [], (), np.nan, False
        x, _ = nnls(*self._weighted(A, b, rows))
        if porosity is None:
            porosity = float(x[-1])
            x = x[:-1]
        available = 1.0 - vcl - porosity
        closed = abs(float(x.sum()) - available) <= CLOSURE_TOLERANCE
        if available <= 1e-9:
            # Clay and porosity fill the rock; rounding puts an exact fill at
            # +1e-17 or -1e-17 depending on the numpy release.
            x = np.zeros_like(x)
        else:
            x = _constrain_mineral_fractions(x, available)
        return minerals, tuple(float(v) for v in x), porosity, closed

    # -- whole curves -------------------------------------------------------------

    def solve(
        self,
        rhob: Union[float, np.ndarray],
        nphi: Optional[Union[float, np.ndarray]] = None,
        vcl:  Optional[Union[float, np.ndarray]] = None,
        phie: Optional[Union[float, np.ndarray]] = None,
        pef:  Optional[Union[float, np.ndarray]] = None,
        dt:   Optional[Union[float, np.ndarray]] = None,
        core_phi: Optional[Union[float, np.ndarray]] = None,
        samples: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve for mineral fractions.

        Parameters
        ----------
        rhob : float or np.ndarray
            Bulk density (g/cm³).
        nphi : float or np.ndarray, optional
            Neutron porosity (v/v); None when unavailable.
        vcl : float or np.ndarray, optional
            Pre-computed clay volume (0-1); 0 when not given.
        phie : float or np.ndarray, optional
            Effective porosity (0-1), used where the sample has NPHI. Where
            it is missing, porosity is solved with the minerals.
        pef, dt : float or np.ndarray, optional
            Photoelectric factor (b/e) and sonic slowness (μs/ft).
        core_phi : float or np.ndarray, optional
            Core porosity (0-1): where porosity is solved, a finite value
            fixes it instead.
        samples : np.ndarray of bool, optional
            Samples to solve; the others are left unsolved. Each sample is
            solved on its own, so the solved ones are as a full solve gives them.

        Returns
        -------
        dict
            Volumes for every mineral (0 where a mineral was not active at
            that depth, NaN where the sample was not solved), the input
            ``vcl``, ``phie`` (the porosity used: the input, or the solved
            one), ``porosity_solved`` (True where porosity was solved or
            fixed by core), ``qc_flag`` (True where not solved) and
            ``solved`` (its inverse).
        """
        rhob = np.atleast_1d(np.asarray(rhob, dtype=float))
        n    = len(rhob)

        def array(values, fill):
            if values is None:
                return np.full(n, fill, dtype=float)
            return np.array(np.broadcast_to(np.atleast_1d(np.asarray(values, dtype=float)), (n,)), dtype=float)

        nphi = array(nphi, np.nan)
        vcl  = array(vcl, 0.0)
        phie = array(phie, np.nan)
        pef_arr = None if pef is None else array(pef, np.nan)
        dt_arr  = None if dt is None else array(dt, np.nan)
        core = array(core_phi, np.nan)

        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef + self.minerals_reduced))
        result = {mineral: np.zeros(n) for mineral in all_minerals}
        result["vcl"]     = vcl.copy()
        result["phie"]    = phie.copy()
        result["qc_flag"] = np.zeros(n, dtype=bool)
        result["porosity_solved"] = np.zeros(n, dtype=bool)

        selected = np.ones(n, dtype=bool) if samples is None else np.broadcast_to(np.asarray(samples, dtype=bool), (n,))
        given = ~(np.isnan(rhob) | np.isnan(nphi) | np.isnan(vcl) | np.isnan(phie)) & selected
        for mineral in all_minerals:
            result[mineral][~given] = np.nan
        result["qc_flag"][~given] = True

        has_pef = np.zeros(n, dtype=bool) if pef_arr is None else ~np.isnan(pef_arr)
        has_dt = np.zeros(n, dtype=bool) if dt_arr is None else ~np.isnan(dt_arr)
        if not self.sonic_without_pef:
            # 4 minerals need at least 4 equations: Unity + RHOB + NPHI + (PEF or DTC).
            has_dt = has_dt & has_pef
        # Samples sharing which logs they have share one system matrix; each is
        # solved on its own.
        for pef_on in (False, True):
            for dt_on in (False, True):
                rows = np.flatnonzero(given & (has_pef == pef_on) & (has_dt == dt_on))
                if rows.size:
                    self._solve_group(rows, rhob, nphi, vcl, phie, pef_arr, dt_arr, pef_on, dt_on, result)

        if self.solve_porosity:
            open_rows = np.flatnonzero(selected & ~given & np.isfinite(rhob) & (rhob > 0.0) & np.isfinite(vcl))
            for i in open_rows:
                pef_i = None if pef_arr is None else _finite(pef_arr[i])
                dt_i = None if dt_arr is None else _finite(dt_arr[i])
                if pef_i is not None and pef_i < 0.0:
                    pef_i = None
                if dt_i is not None and dt_i <= 0.0:
                    dt_i = None
                minerals, volumes, porosity, closed = self._solve_porosity_sample(
                    float(rhob[i]), _finite(nphi[i]), float(vcl[i]), pef_i, dt_i, _finite(core[i]))
                if not minerals or not closed:
                    continue
                for mineral in all_minerals:
                    result[mineral][i] = 0.0
                for mineral, volume in zip(minerals, volumes):
                    result[mineral][i] = volume
                result["phie"][i] = porosity
                result["qc_flag"][i] = False
                result["porosity_solved"][i] = True

        result["solved"] = ~result["qc_flag"]
        return result

    def _solve_group(self, rows, rhob, nphi, vcl, phie, pef_arr, dt_arr, pef_on, dt_on, result) -> None:
        """Solve the samples ``rows``, which have the same logs and a given PHIE, into ``result``."""
        active_minerals = self.minerals if (pef_on or dt_on) else self.minerals_no_pef
        m = [self.mineral_props[name] for name in active_minerals]
        fl, cl = self.fluid_props, self.clay_props
        vcl_s = np.clip(vcl[rows], 0.0, 1.0)
        phie_s = np.clip(phie[rows], 0.0, 1.0)

        A_rows, b_rows, row_names = [], [], []

        def add_row(row, mineral_vals, cl_val, fl_val, log_val):
            A_rows.append(mineral_vals)
            b_rows.append(log_val - vcl_s * cl_val - phie_s * fl_val)
            row_names.append(row)

        add_row("unity", [1.0] * len(active_minerals), cl_val=1.0, fl_val=1.0, log_val=1.0)
        add_row("rhob", [mi["density"] for mi in m], cl_val=cl["density"], fl_val=fl["density"], log_val=rhob[rows])
        add_row("nphi", [mi["nphi"] for mi in m], cl_val=cl["nphi"], fl_val=fl["nphi"], log_val=nphi[rows])
        if pef_on:
            pef_s = pef_arr[rows]
            add_row("u", [mi["u"] for mi in m], cl_val=cl["u"], fl_val=fl["u"], log_val=pef_s * rhob[rows])
        if dt_on:
            dt_s = dt_arr[rows]
            add_row("dt", [mi["dt"] for mi in m], cl_val=cl["dt"], fl_val=fl["dt"], log_val=dt_s)

        A = np.array(A_rows)
        B = np.ascontiguousarray(np.array(b_rows).T)  # one right-hand side per sample
        if self.row_uncertainty is not None:
            weights = 1.0 / np.array([self.row_uncertainty[row] for row in row_names])
            A, B = A * weights[:, None], B * weights
        X = nnls_many(A, B)

        available = 1.0 - vcl_s - phie_s
        closed = ~(np.abs(X.sum(axis=1) - available) > CLOSURE_TOLERANCE)
        # _constrain_mineral_fractions, sample by sample.
        X = np.clip(X, 0.0, np.maximum(available, 0.0)[:, None])
        total = np.nansum(X, axis=1)
        scale = (total > available) & (total > 0.0)
        X[scale] *= (available[scale] / total[scale])[:, None]
        X[~closed] = np.nan
        for k, mineral in enumerate(active_minerals):
            result[mineral][rows] = X[:, k]
        result["qc_flag"][rows[~closed]] = True

    # -- auditing ---------------------------------------------------------------------

    def explain_sample(self, well: "Well" = None, index: int = 0, *, depth=None,
                       rhob=None, nphi=None, vcl=None, phie=None, pef=None, dt=None,
                       core_phi=None) -> Dict[str, object]:
        """Return the exact linear system solved at one depth, for auditing and teaching.

        Pass a ``well`` (uses sample ``index``, or the nearest sample to ``depth``)
        or the log values directly. The returned dict has the labelled response
        matrix ``A`` and right-hand side ``b`` (before the rows are divided by
        ``row_uncertainty``), the solution ``x``, the row and column labels,
        and the porosity used, so you can see precisely how a mineral solution
        was produced.
        """
        if well is not None and depth is not None:
            dep = np.asarray(well.depth, dtype=float)
            index = int(np.nanargmin(np.abs(dep - float(depth))))
        if well is not None:
            def pick(*names):
                return next((np.asarray(well.get(n).data, dtype=float)[index]
                             for n in names if well.get(n) is not None), None)
            rhob = pick("RHOB") if rhob is None else rhob
            nphi = pick("NPHI", "PHIN") if nphi is None else nphi
            vcl = pick("VCL", "VCLAY", "VSH") if vcl is None else vcl
            phie = pick("PHIE", "PHIT") if phie is None else phie
            pef = pick("PEF") if pef is None else pef
            dt = pick("DTC", "DT") if dt is None else dt
            if core_phi is None:
                core_phi = pick("PHIE_CC", "PHIT_CC", "CPOR", "CPOR_CC", "PHIC")
        if rhob is None or not np.isfinite(rhob):
            raise ValueError("explain_sample needs a finite rhob")
        rhob = float(rhob)
        vcl = float(np.clip(_finite(vcl) or 0.0, 0.0, 1.0))
        nphi, phie, pef, dt, core_phi = (_finite(v) for v in (nphi, phie, pef, dt, core_phi))

        if nphi is not None and phie is not None:
            porosity: Optional[float] = float(np.clip(phie, 0.0, 1.0))
            minerals = self._fixed_minerals(pef, dt)
            rows = ["unity", "rhob", "nphi"]
            if pef is not None:
                rows.append("u")
            if dt is not None and (pef is not None or self.sonic_without_pef):
                rows.append("dt")
            source = "PHIE given"
        else:
            if not self.solve_porosity:
                raise ValueError("porosity is not solved by this solver (solve_porosity=False) and PHIE "
                                 "or NPHI is missing at this sample")
            porosity = None if core_phi is None else float(np.clip(core_phi, 0.0, 1.0 - vcl))
            rows = self._rows(nphi, pef, dt)
            minerals = self._solved_minerals(rows, porosity is not None)
            source = "core porosity" if porosity is not None else "solved"
            if not minerals:
                raise ValueError("not enough equations to resolve two minerals at this sample "
                                 "(porosity solved: RHOB, closure and PEF or NPHI at least)")
        A, b = self._system(minerals, rows, rhob, nphi, vcl, pef, dt, porosity)
        x, residual = nnls(*self._weighted(A, b, rows))
        return {
            "minerals": list(minerals), "row_labels": [_ROW_LABELS[row] for row in rows],
            "columns": list(minerals) + ([] if porosity is not None else ["porosity"]),
            "A": A, "b": b, "x": x, "residual": float(residual),
            "row_uncertainty": (None if self.row_uncertainty is None
                                else [self.row_uncertainty[row] for row in rows]),
            "vcl": vcl, "porosity": porosity if porosity is not None else float(x[-1]),
            "porosity_source": source,
        }

    def add_to_well(self, well: "Well") -> "Well":
        """
        Add mineral volume curves to the well.

        Requires RHOB and VCL on the well, with NPHI/PHIN or PEF. PHIE (or
        PHIT) is used where the sample has NPHI; elsewhere porosity is solved
        with the minerals.

        Parameters
        ----------
        well : Well
            Well object

        Returns
        -------
        Well
            Well with VQTZ, VCAL, VDOL, VANH and the MMQC flag added
        """
        rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
        nphi, _, _ = get_curve_data_depthwise(well, ["NPHI", "PHIN"])
        vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        pef, _, _ = get_curve_data_depthwise(well, ["PEF"])
        dt, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])

        missing = [name for name, data in zip(["RHOB", "VCL/VCLAY/VSH"], [rhob, vcl]) if data is None]
        if nphi is None and pef is None:
            missing.append("NPHI/PHIN or PEF")
        if missing:
            raise ValueError(f"Missing required curves: {missing}")

        result = self.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)

        curve_names = {
            "quartz":    "VQTZ",
            "calcite":   "VCAL",
            "dolomite":  "VDOL",
            "anhydrite": "VANH",
        }
        for mineral, values in result.items():
            if mineral in ("vcl", "phie", "qc_flag", "solved", "porosity_solved"):
                continue
            curve_name = curve_names.get(mineral, f"V{mineral.upper()[:4]}")
            well.add_curve(
                curve_name, values, unit="v/v",
                description=f"Volume {mineral}",
            )

        well.add_curve(
            "MMQC", result["qc_flag"].astype(float), unit="",
            description="Multi-mineral closure QC flag (1 = not solved)",
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
    Convenience wrapper for a 2-mineral solve.

    It routes through ``MultiMineralSolver``; VCL defaults to zero, and
    without ``phie`` (or ``nphi``) porosity is solved with the minerals.
    """
    solver = MultiMineralSolver(
        minerals=list(minerals),
        minerals_no_pef=list(minerals),
        minerals_reduced=list(minerals),
        **kwargs,
    )
    return solver.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)


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
    Convenience wrapper for a 3-mineral solve.

    It routes through ``MultiMineralSolver``; VCL defaults to zero, and
    without ``phie`` (or ``nphi``) porosity is solved with the minerals.
    """
    solver = MultiMineralSolver(
        minerals=list(minerals),
        minerals_no_pef=list(minerals),
        minerals_reduced=list(minerals),
        **kwargs,
    )
    return solver.solve(rhob, nphi, vcl, phie, pef=pef, dt=dt)

