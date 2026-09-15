"""
Multi-mineral solver for lithology determination.

This module provides solvers for determining mineral fractions
from multiple log measurements using simultaneous equations.
"""

from typing import Optional, Tuple, Dict, Union, Sequence
import numpy as np
from scipy.optimize import nnls
from petrophys.model.well import Well

from petrophys.interpretation.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)
from petrophys._internal.analysis_helpers import get_curve_data_depthwise


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
        # Derive the U endpoint with the same rule as the measured log term
        # (U = PEF·ρ, see _u) so the solver's U row is internally consistent.
        # The hardcoded literature "u" values differ from PEF·ρ by up to ~0.1
        # and otherwise inject a small spurious-mineral bias on pure lithologies.
        "u":       _u(p["pef"], p["density"]),
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
    if total > 0.0:
        constrained *= float(max(available_fraction, 0.0)) / total
    elif available_fraction <= 0.0:
        constrained[:] = 0.0
    else:
        constrained[:] = np.nan
    return constrained


class MultiMineralSolver:
    """
    Multi-mineral solver for complex lithology.

    This class solves for mineral fractions using multiple
    log measurements simultaneously.

    Vcl (from VclModule) must be pre-computed before calling this solver and is
    subtracted from the RHS of every equation. Porosity is NOT taken as an input
    (a pre-computed PHIE depends on the assumed matrix, which would bias the
    result); it is solved as an extra unknown alongside the minerals, so the
    mineral solution is independent of any matrix guess. The ``phie`` argument is
    accepted for backward compatibility but is not used in the solve; the solved
    porosity is returned in ``result["phie"]``.

    The active mineral set and equations are determined per depth from the
    equations available at that sample. Unity and RHOB are always used; NPHI,
    PEF, and DTC are optional discriminator rows. The full mineral set is used
    when enough independent rows are available, otherwise the solver falls
    back to the reduced mineral set.

    PEF is preferred over DTC for mineral discrimination because U = PEF × ρ
    has a much wider spread across carbonates (quartz: 4.77, dolomite: 8.90,
    calcite: 13.82 b/cm³) and is the only tool that reliably separates
    anhydrite from dolomite.

    Tool Response Equations (linear mixing), with porosity φ as an unknown:
        Unity:   V1 + ... + Vn + φ          = 1    - Vcl
        Density: V1·ρ1 + ... + Vn·ρn + φ·ρfl = ρlog - Vcl·ρcl
        Neutron: V1·N1 + ... + Vn·Nn + φ·Nfl = Nlog - Vcl·Ncl
        U:       V1·U1 + ... + Vn·Un + φ·Ufl = Ulog - Vcl·Ucl
        Sonic:   V1·D1 + ... + Vn·Dn + φ·Dfl = Dlog - Vcl·Dcl

    where U = PEF × ρ (volumetric photoelectric cross-section). Solving φ as an
    unknown (rather than subtracting a pre-computed PHIE) is what makes the
    mineral result independent of the assumed matrix.

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
        phie: Optional[float],
        pef: Optional[float],
        dt: Optional[float],
        phi_fixed: Optional[float] = None,
    ) -> Tuple[list, tuple, float, bool, float]:
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
        # Porosity handling:
        #   - if a known (core) porosity is supplied, fix it -> the matrix density
        #     is determined, which resolves quartz/calcite even without PEF;
        #   - otherwise solve porosity as an unknown so the result does not depend
        #     on a matrix-biased PHIE.
        # Either way only Vcl (matrix-independent, GR-driven) is on the RHS.
        use_fixed_phi = phi_fixed is not None and np.isfinite(phi_fixed)
        if use_fixed_phi:
            phi_fixed = float(np.clip(phi_fixed, 0.0, 1.0 - vcl))

        active_minerals = self._active_minerals(nphi, pef, dt, use_fixed_phi)
        if not active_minerals:
            return [], tuple(), np.nan, True, np.nan

        A_arr, b_arr, _row_labels, _columns = self._assemble_sample(
            active_minerals, rhob, nphi, vcl, pef, dt,
            phi_fixed if use_fixed_phi else None,
        )
        if np.linalg.matrix_rank(A_arr) < A_arr.shape[1]:
            return [], tuple(), np.nan, True, np.nan
        x, residual = nnls(A_arr, b_arr)

        if use_fixed_phi:
            mineral_x = x
            phi_solved = phi_fixed
        else:
            # Scale the complete non-negative solution to the non-clay volume.
            # This preserves all relative NNLS contributions while guaranteeing
            # Vcl + porosity + minerals = 1 at every solved sample.
            raw_total = float(np.sum(x))
            closure_target = max(1.0 - vcl, 0.0)
            closure_error = abs(raw_total - closure_target)
            if raw_total > 0.0:
                x = x * closure_target / raw_total
            elif closure_target > 0.0:
                return [], tuple(), float(residual), True, np.nan
            mineral_x = x[:-1]
            phi_solved = float(x[-1])

        available_fraction = max(1.0 - vcl - phi_solved, 0.0)
        if use_fixed_phi:
            closure_error = abs(float(np.sum(mineral_x)) - available_fraction)
        poor_fit = closure_error > self.closure_tolerance

        # Always normalize, never discard a solution just for closure error.
        # The poor_fit flag and residual give the caller full quality info.
        mineral_x = _constrain_mineral_fractions(mineral_x, available_fraction)
        if not np.all(np.isfinite(mineral_x)):
            return [], tuple(), float(residual), True, np.nan

        return (active_minerals, tuple(float(v) for v in mineral_x),
                float(residual), poor_fit, phi_solved)

    def _active_minerals(self, nphi, pef, dt, use_fixed_phi: bool) -> list:
        """Pick the mineral set the available equations can resolve at a sample."""
        n_porosity_unknowns = 0 if use_fixed_phi else 1
        available_equations = 2 + sum(v is not None for v in (nphi, pef, dt))
        if available_equations >= len(self.minerals) + n_porosity_unknowns:
            return self.minerals
        if available_equations >= len(self.minerals_no_pef) + n_porosity_unknowns:
            return self.minerals_no_pef
        return []

    def _assemble_sample(self, active_minerals, rhob, nphi, vcl, pef, dt, phi_fixed):
        """Build the labelled response system A x = b for one sample.

        Returns (A, b, row_labels, column_labels). Shared by the solver and by
        :meth:`explain_sample`, so what you inspect is exactly what is solved.
        """
        use_fixed_phi = phi_fixed is not None and np.isfinite(phi_fixed)
        vcl = float(np.clip(vcl, 0.0, 1.0))
        m = [self.mineral_props[name] for name in active_minerals]
        fl, cl = self.fluid_props, self.clay_props
        A_rows, b_rows, labels = [], [], []

        def add_row(label, key, log_val):
            mineral_vals = [1.0] * len(m) if key is None else [mi[key] for mi in m]
            fl_val = 1.0 if key is None else fl[key]
            cl_val = 1.0 if key is None else cl[key]
            if use_fixed_phi:
                A_rows.append(list(mineral_vals))
                b_rows.append(log_val - vcl * cl_val - phi_fixed * fl_val)
            else:
                A_rows.append(list(mineral_vals) + [fl_val])
                b_rows.append(log_val - vcl * cl_val)
            labels.append(label)

        add_row("Unity", None, 1.0)
        add_row("Density (RHOB)", "density", rhob)
        if nphi is not None:
            add_row("Neutron (NPHI)", "nphi", nphi)
        if pef is not None:
            add_row("U = PEF·ρ", "u", _u(pef, rhob))
        if dt is not None:
            add_row("Sonic (DT)", "dt", dt)

        columns = list(active_minerals) + ([] if use_fixed_phi else ["porosity"])
        return np.array(A_rows, dtype=float), np.array(b_rows, dtype=float), labels, columns

    def explain_sample(self, well: "Well" = None, index: int = 0, *, depth=None,
                       rhob=None, nphi=None, vcl=None, pef=None, dt=None,
                       core_phi=None) -> Dict[str, object]:
        """Return the exact linear system solved at one depth, for auditing and teaching.

        Pass a ``well`` (uses sample ``index``, or the nearest sample to ``depth``)
        or the log values directly. The returned dict has the labelled response
        matrix ``A``, right-hand side ``b``, the NNLS solution ``x``, the
        ``residual``, and the row/column labels, so you can see precisely how a
        mineral solution was produced.
        """
        if well is not None and depth is not None:
            dep = np.asarray(well.depth, dtype=float)
            index = int(np.nanargmin(np.abs(dep - float(depth))))
        if well is not None:
            pick = lambda *names: next((np.asarray(well.get(n).data, dtype=float)[index]
                                        for n in names if well.get(n) is not None), None)
            rhob = pick("RHOB") if rhob is None else rhob
            nphi = pick("NPHI", "PHIN") if nphi is None else nphi
            vcl = pick("VCL", "VCLAY", "VSH") if vcl is None else vcl
            pef = pick("PEF") if pef is None else pef
            dt = pick("DTC", "DT") if dt is None else dt
            if core_phi is None:
                core_phi = pick("PHIC", "CPOR")
        vcl = 0.0 if vcl is None or not np.isfinite(vcl) else float(vcl)
        phi_fixed = float(core_phi) if core_phi is not None and np.isfinite(core_phi) else None
        for name, val in (("rhob", rhob),):
            if val is None or not np.isfinite(val):
                raise ValueError(f"explain_sample needs a finite {name}")
        nphi = None if nphi is None or not np.isfinite(nphi) else float(nphi)
        pef = None if pef is None or not np.isfinite(pef) else float(pef)
        dt = None if dt is None or not np.isfinite(dt) else float(dt)

        use_fixed_phi = phi_fixed is not None
        active = self._active_minerals(nphi, pef, dt, use_fixed_phi)
        if not active:
            raise ValueError("not enough equations to resolve any mineral at this sample")
        A, b, row_labels, columns = self._assemble_sample(
            active, float(rhob), nphi, vcl, pef, dt, phi_fixed)
        x, residual = nnls(A, b)
        return {
            "minerals": list(active), "row_labels": row_labels, "columns": columns,
            "A": A, "b": b, "x": x, "residual": float(residual),
            "vcl": vcl, "porosity_solved": (phi_fixed if use_fixed_phi else float(x[-1])),
        }

    def solve(
        self,
        rhob: Union[float, np.ndarray],
        nphi: Optional[Union[float, np.ndarray]],
        vcl:  Union[float, np.ndarray],
        phie: Optional[Union[float, np.ndarray]] = None,
        pef:  Optional[Union[float, np.ndarray]] = None,
        dt:   Optional[Union[float, np.ndarray]] = None,
        core_phi: Optional[Union[float, np.ndarray]] = None,
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
        phie : float or np.ndarray, optional
            Accepted for backward compatibility. The solver derives porosity
            jointly with mineral volumes and does not require this input.
        pef : float or np.ndarray, optional
            Photoelectric factor (b/e); omit if curve absent
        dt : float or np.ndarray, optional
            Sonic slowness (μs/ft); omit if curve absent
        core_phi : float or np.ndarray, optional
            Independently constrained porosity (fraction). Finite samples fix
            porosity; missing samples leave it as an unknown.

        Returns
        -------
        dict
            Dictionary with mineral fractions, vcl, phie, residual, qc_flag,
            solved, and underdetermined. Unsolved mineral fractions are NaN.
            Underdetermined marks samples with too few response equations;
            it does not mark every failed fit.
            All minerals from both self.minerals and self.minerals_no_pef
            are present as keys; depths where a mineral was not in the
            active set are filled with 0.
        """
        raw_arrays = {
            "rhob": np.atleast_1d(np.asarray(rhob, dtype=float)),
            "vcl": np.atleast_1d(np.asarray(vcl, dtype=float)),
        }
        if phie is not None:
            raw_arrays["phie"] = np.atleast_1d(np.asarray(phie, dtype=float))
        if nphi is not None:
            raw_arrays["nphi"] = np.atleast_1d(np.asarray(nphi, dtype=float))
        if pef is not None:
            raw_arrays["pef"] = np.atleast_1d(np.asarray(pef, dtype=float))
        if dt is not None:
            raw_arrays["dt"] = np.atleast_1d(np.asarray(dt, dtype=float))
        if core_phi is not None:
            raw_arrays["core_phi"] = np.atleast_1d(np.asarray(core_phi, dtype=float))

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
        phie_arr = _broadcast("phie", raw_arrays["phie"]) if phie is not None else None
        pef_arr = _broadcast("pef", raw_arrays["pef"]) if pef is not None else None
        dt_arr = _broadcast("dt", raw_arrays["dt"]) if dt is not None else None
        core_phi_arr = _broadcast("core_phi", raw_arrays["core_phi"]) if core_phi is not None else None

        # All possible minerals across both sets
        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))

        result = {mineral: np.zeros(n) for mineral in all_minerals}
        result["vcl"]      = vcl.copy()
        result["phie"]     = np.full(n, np.nan, dtype=float)
        result["residual"] = np.zeros(n)
        result["qc_flag"]  = np.zeros(n, dtype=bool)
        result["solved"]   = np.zeros(n, dtype=bool)
        result["underdetermined"] = np.zeros(n, dtype=bool)

        for i in range(n):
            if not np.all(np.isfinite([rhob[i], vcl[i]])) or rhob[i] <= 0.0:
                for mineral in all_minerals:
                    result[mineral][i] = np.nan
                result["residual"][i] = np.nan
                result["qc_flag"][i]  = True
                continue

            nphi_i = float(nphi_arr[i]) if nphi_arr is not None else None
            pef_i = float(pef_arr[i]) if pef_arr is not None else None
            dt_i  = float(dt_arr[i])  if dt_arr  is not None else None
            core_phi_i = float(core_phi_arr[i]) if core_phi_arr is not None else None

            if nphi_i is not None and not np.isfinite(nphi_i):
                nphi_i = None
            if pef_i is not None and (not np.isfinite(pef_i) or pef_i < 0.0):
                pef_i = None
            if dt_i is not None and (not np.isfinite(dt_i) or dt_i <= 0.0):
                dt_i  = None
            if core_phi_i is not None and not np.isfinite(core_phi_i):
                core_phi_i = None

            result["underdetermined"][i] = not self._active_minerals(
                nphi_i, pef_i, dt_i, core_phi_i is not None
            )

            active_minerals, fractions, residual, poor_fit, phi_solved = self._solve_sample(
                rhob[i], nphi_i, vcl[i],
                float(phie_arr[i]) if phie_arr is not None else None,
                pef_i, dt_i, phi_fixed=core_phi_i
            )

            if not active_minerals:
                for mineral in all_minerals:
                    result[mineral][i] = np.nan
                result["residual"][i] = residual
                result["qc_flag"][i] = True
                continue

            for mineral, v in zip(active_minerals, fractions):
                result[mineral][i] = v

            # Report the solver-derived (matrix-independent) porosity.
            result["phie"][i] = phi_solved
            result["residual"][i] = residual
            result["qc_flag"][i]  = poor_fit
            result["solved"][i] = True

        return result

    def add_to_well(self, well: "Well") -> "Well":
        """
        Add lithology curves to well.

        Requires RHOB plus VCL on the well object. At least one
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
        pef, _, _ = get_curve_data_depthwise(well, ["PEF"])
        dt, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])

        missing = [
            name for name, data in zip(
                ["RHOB", "VCL/VCLAY/VSH"],
                [rhob, vcl],
            )
            if data is None
        ]
        if nphi is None and pef is None and dt is None:
            missing.append("NPHI/PHIN, PEF, or DTC/DT")
        if missing:
            raise ValueError(f"Missing required curves: {missing}")

        result = self.solve(
            rhob, nphi, vcl,
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
            if mineral in ("vcl", "phie", "residual", "qc_flag", "solved", "underdetermined"):
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
