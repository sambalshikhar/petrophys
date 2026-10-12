"""
Main lithology determination module.

This module provides the LithologyModule class for comprehensive
lithology analysis from well logs.
"""
from typing import TYPE_CHECKING, Optional, List
from dataclasses import dataclass
import numpy as np

from petrophys.interpretation.lithology.multi_mineral_solver_v1 import MultiMineralSolverV1
from petrophys.interpretation.lithology.multi_mineral_solver import (
    MultiMineralSolver,
)
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class LithologyResult:
    """
    Result container for lithology analysis.

    Attributes
    ----------
    v_quartz : np.ndarray
        Quartz volume fraction
    v_calcite : np.ndarray
        Calcite volume fraction
    v_dolomite : np.ndarray
        Dolomite volume fraction
    v_anhydrite : np.ndarray
        Anhydrite volume fraction
    v_clay : np.ndarray
        Clay volume fraction (from VCL)
    lithology_code : np.ndarray
        Numeric lithology codes
    lithology_name : np.ndarray
        Lithology names
    """

    v_quartz: np.ndarray
    v_calcite: np.ndarray
    v_dolomite: np.ndarray
    v_anhydrite: np.ndarray
    v_clay: np.ndarray
    lithology_code: np.ndarray
    lithology_name: np.ndarray
    qc_flag: np.ndarray
    solved: np.ndarray
    # Samples recovered with the constrained quartz-calcite fallback.  This is
    # separate from ``solved`` so callers can audit the modelling assumption.
    reduced_qc_fallback: Optional[np.ndarray] = None
    # RHOmaa-Umaa automated cross-check (None when PEF is unavailable).
    rhomaa: Optional[np.ndarray] = None
    umaa: Optional[np.ndarray] = None
    litho_disagree: Optional[np.ndarray] = None
    mineral_selection: Optional[dict] = None


class LithologyModule:
    """
    Lithology determination from well logs.

    This module combines multi-mineral analysis with clay volume
    to determine complete lithology fractions.

    Parameters
    ----------
    minerals : list, optional
        Primary minerals to solve for
    include_clay : bool, optional
        Include clay from VCL (default: True)
    lithology : {"auto", "clastic", "carbonate"}, optional
        Geological constraint for a reduced mineral set. Auto uses an input
        lithology log or core-constrained matrix density with at least 80%
        agreement. Carbonate samples with limited logs use calcite and dolomite;
        the full mineral set remains available when there are enough equations.
    solver_version : {1, 2}, optional
        Mineral solver version (default: 2). Version 2 is
        :class:`MultiMineralSolver`, which solves porosity jointly with the
        minerals. Version 1 is :class:`MultiMineralSolverV1`, the original
        solver that takes PHIE as a fixed input and leaves samples unsolved
        when the closure gate fails; the reduced-mineral selection above
        applies to version 2 only.
    solver_options : dict, optional
        Further keyword arguments for the solver class, e.g.
        ``{"row_uncertainty": None}`` for the unscaled version 1 solve.

    Examples
    --------
    >>> litho = LithologyModule()
    >>> result = litho.calculate(well)
    >>> well["VQUAR"] = result.v_quartz
    """

    # Lithology codes
    LITHOLOGY_CODES = {
        "sandstone": 1,
        "limestone": 2,
        "dolomite": 3,
        "shale": 4,
        "anhydrite": 5,
        "salt": 6,
        "coal": 7,
        "mixed": 8,
        "unknown": 0,
    }

    def __init__(
        self,
        minerals: List[str] = None,
        include_clay: bool = True,
        lithology: str = "auto",
        solver_version: int = 2,
        solver_options: Optional[dict] = None,
    ):
        self._custom_minerals = minerals is not None
        if lithology not in {"auto", "clastic", "carbonate"}:
            raise ValueError("lithology must be auto, clastic, or carbonate")
        if solver_version not in (1, 2):
            raise ValueError("solver_version must be 1 or 2")
        self.lithology = lithology
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]

        self.minerals = minerals
        self.include_clay = include_clay
        self.solver_version = solver_version
        self.solver_options = dict(solver_options or {})
        solver_class = MultiMineralSolverV1 if solver_version == 1 else MultiMineralSolver
        self.solver = solver_class(minerals=minerals, **self.solver_options)

    def _select_solver(self, well):
        """Choose a reduced mineral set from independent geological evidence."""
        selection = {
            "lithology": self.lithology,
            "source": "configured" if self.lithology != "auto" else "unconstrained",
            "full_minerals": list(self.solver.minerals),
            "reduced_minerals": list(self.solver.minerals_no_pef),
        }
        if type(self.solver) is not MultiMineralSolver or self._custom_minerals:
            selection["source"] = "custom_solver"
            return self.solver, selection
        family = self.lithology
        if family == "auto":
            from .matrix_inference import resolve_from_core_matrix, resolve_from_lithology_log

            resolution = (resolve_from_lithology_log(well, warn=False)
                          or resolve_from_core_matrix(well, warn=False))
            if resolution is not None:
                selection.update(
                    source=resolution.method, agreement=resolution.confidence,
                    supporting_samples=resolution.n,
                )
                if resolution.confidence >= 0.8:
                    family = resolution.lithology
        selection["lithology"] = family
        if family == "carbonate":
            selection["reduced_minerals"] = ["calcite", "dolomite"]
            return MultiMineralSolver(
                minerals=self.minerals,
                minerals_no_pef=selection["reduced_minerals"],
                **self.solver_options,
            ), selection
        return self.solver, selection

    @staticmethod
    def _screen_pef(pef: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Remove physically implausible PEF values before solving."""
        if pef is None:
            return None
        pef = np.asarray(pef, dtype=float).copy()
        pef[(pef < 0.0) | (pef > 12.0)] = np.nan
        return pef

    @staticmethod
    def _screen_dtc(dtc: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Remove implausible sonic values before solving."""
        if dtc is None:
            return None
        dtc = np.asarray(dtc, dtype=float).copy()
        dtc[(dtc < 35.0) | (dtc > 250.0)] = np.nan
        return dtc

    # --- Volume-based classification (primary path) ---------------------------
    # Clay fraction (whole rock) above which the sample is called shale.
    SHALE_CLAY_CUT = 0.40
    # Minimum dominance of one mineral within the clay-free matrix to name the rock.
    MATRIX_DOMINANCE = 0.50
    # Quartz / calcite / dolomite end-points in (RHOmaa, Umaa) space for the
    # RHOmaa-Umaa cross-check (Schlumberger chartbook values).
    _QCD_ENDPOINTS = {
        "sandstone": (2.65, 4.8),
        "limestone": (2.71, 13.8),
        "dolomite": (2.87, 9.0),
    }

    def _classify_from_volumes(
        self,
        vcl: float,
        v_quartz: float,
        v_calcite: float,
        v_dolomite: float,
        v_anhydrite: float,
        rhob: float,
        nphi: float,
        gr: float,
    ) -> str:
        """Classify lithology from the solved mineral volumes (single source of truth).

        The solver returns whole-rock matrix fractions (clay- and porosity-free),
        so lithology is the dominant grain component, with a clay cut for shale and
        a dominance floor that routes ambiguous mixtures to ``mixed``. Coal and
        halite are not in the solver's mineral set, so they keep an unambiguous
        physical guard (very low density with a distinctive neutron response).
        This is the primary classifier; ``_classify_sample`` is used only when the
        solver could not run (too few curves).
        """
        if np.isfinite(rhob) and np.isfinite(nphi):
            if rhob < 1.95 and nphi > 0.35:
                return "coal"
            if rhob < 2.25 and nphi < 0.05 and (not np.isfinite(gr) or gr < 35):
                return "salt"

        clay = vcl if np.isfinite(vcl) else 0.0
        grains = {
            "sandstone": v_quartz if np.isfinite(v_quartz) else 0.0,
            "limestone": v_calcite if np.isfinite(v_calcite) else 0.0,
            "dolomite": v_dolomite if np.isfinite(v_dolomite) else 0.0,
            "anhydrite": v_anhydrite if np.isfinite(v_anhydrite) else 0.0,
        }
        # The solver can use anhydrite as a high-density sink; only trust an
        # anhydrite call where the density/neutron signature actually supports it
        # (anhydrite ~2.96 g/cc, neutron ~0), otherwise demote it from the vote.
        anhydrite_confirmed = (
            np.isfinite(rhob) and rhob > 2.85 and (not np.isfinite(nphi) or nphi < 0.12)
        )
        if not anhydrite_confirmed:
            grains["anhydrite"] = 0.0
        matrix_total = sum(grains.values())
        if clay <= 0.0 and matrix_total <= 1e-6:
            return "unknown"
        # Clay cut on the whole-rock clay fraction (standard Vsh-style threshold).
        if clay >= self.SHALE_CLAY_CUT or matrix_total <= 1e-6:
            return "shale"

        dominant = max(grains, key=grains.get)
        if grains[dominant] / matrix_total >= self.MATRIX_DOMINANCE:
            return dominant
        # No single mineral dominates: a combined carbonate still names a carbonate.
        if (grains["limestone"] + grains["dolomite"]) / matrix_total >= self.MATRIX_DOMINANCE:
            return "dolomite" if grains["dolomite"] > grains["limestone"] else "limestone"
        return "mixed"

    def _classify_codes_from_volumes(self, vcl, v_quartz, v_calcite, v_dolomite,
                                     v_anhydrite, rhob, nphi, gr) -> np.ndarray:
        """Vectorized form of :meth:`_classify_from_volumes` returning lithology
        codes for whole arrays. Byte-identical to the per-sample version; used to
        avoid a Python loop over every depth (verified against the scalar path)."""
        C = self.LITHOLOGY_CODES
        f = np.isfinite
        n = np.asarray(vcl).shape[0]
        clay = np.where(f(vcl), vcl, 0.0)
        q = np.where(f(v_quartz), v_quartz, 0.0)
        c = np.where(f(v_calcite), v_calcite, 0.0)
        d = np.where(f(v_dolomite), v_dolomite, 0.0)
        a0 = np.where(f(v_anhydrite), v_anhydrite, 0.0)
        fin_rb, fin_np, fin_g = f(rhob), f(nphi), f(gr)

        coal = fin_rb & fin_np & (rhob < 1.95) & (nphi > 0.35)
        salt = fin_rb & fin_np & (rhob < 2.25) & (nphi < 0.05) & (~fin_g | (gr < 35))

        anh_ok = fin_rb & (rhob > 2.85) & (~fin_np | (nphi < 0.12))
        a = np.where(anh_ok, a0, 0.0)
        mt = q + c + d + a

        stack = np.vstack([q, c, d, a])         # order = sandstone, limestone, dolomite, anhydrite
        dom_idx = np.argmax(stack, axis=0)       # ties -> first, matches max(dict)
        dom_val = stack[dom_idx, np.arange(n)]
        dom_code = np.array([C["sandstone"], C["limestone"], C["dolomite"], C["anhydrite"]])[dom_idx]
        safe_mt = np.where(mt > 1e-12, mt, 1.0)
        dom_frac = dom_val / safe_mt
        carb_frac = (c + d) / safe_mt
        carb_code = np.where(d > c, C["dolomite"], C["limestone"])

        base = ~coal & ~salt
        conds = [
            coal,
            salt & ~coal,
            base & (clay <= 0.0) & (mt <= 1e-6),
            base & ((clay >= self.SHALE_CLAY_CUT) | (mt <= 1e-6)),
            base & (mt > 1e-6) & (dom_frac >= self.MATRIX_DOMINANCE),
            base & (mt > 1e-6) & (dom_frac < self.MATRIX_DOMINANCE) & (carb_frac >= self.MATRIX_DOMINANCE),
        ]
        choices = [
            np.full(n, C["coal"]), np.full(n, C["salt"]), np.full(n, C["unknown"]),
            np.full(n, C["shale"]), dom_code, carb_code,
        ]
        return np.select(conds, choices, default=C["mixed"]).astype(int)

    @staticmethod
    def _rhomaa_umaa(rhob: float, nphi: float, pef: float):
        """Apparent matrix density and apparent matrix volumetric PE (RHOmaa, Umaa).

        Independent deterministic transform behind the RHOmaa-Umaa crossplot; used
        here as an automated cross-check (no human/plot required). Needs RHOB, NPHI
        and PEF; returns (nan, nan) otherwise.
        """
        if not (np.isfinite(rhob) and np.isfinite(nphi) and np.isfinite(pef)):
            return np.nan, np.nan
        rho_f, u_f = 1.0, 0.398  # water end-points
        phi_d = (2.71 - rhob) / (2.71 - rho_f)         # limestone-referenced density porosity
        phi_ta = float(np.clip(0.5 * (phi_d + nphi), 0.0, 0.6))  # apparent total porosity
        denom = 1.0 - phi_ta
        if denom <= 1e-6:
            return np.nan, np.nan
        rho_e = (rhob + 0.1883) / 1.0704
        u = pef * rho_e
        rhomaa = (rhob - phi_ta * rho_f) / denom
        umaa = (u - phi_ta * u_f) / denom
        return rhomaa, umaa

    def _ternary_qcd_lithology(self, rhomaa: float, umaa: float):
        """Dominant of quartz/calcite/dolomite from the (RHOmaa, Umaa) ternary, or None."""
        if not (np.isfinite(rhomaa) and np.isfinite(umaa)):
            return None
        labels = list(self._QCD_ENDPOINTS)
        a = np.array([
            [self._QCD_ENDPOINTS[name][0] for name in labels],
            [self._QCD_ENDPOINTS[name][1] for name in labels],
            [1.0, 1.0, 1.0],
        ])
        b = np.array([rhomaa, umaa, 1.0])
        try:
            frac = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        # Point well outside the QCD triangle -> unreliable, no verdict.
        if np.any(frac < -0.15) or np.any(frac > 1.15):
            return None
        return labels[int(np.argmax(frac))]

    def _classify_sample(
        self,
        rhob: float,
        nphi: float,
        gr: float,
        dtc: float,
        vcl: float,
        v_quartz: float,
        v_calcite: float,
        v_dolomite: float,
        v_anhydrite: float,
    ) -> str:
        """
        Classify broad lithology from logs and solved mineral fractions.

        The solver provides mineral volumes; raw curves are then used to keep
        petrophysical families aligned with field-observed lithologies.
        """
        if not np.isfinite(rhob):
            return "unknown"

        carbonate_volume = v_calcite + v_dolomite
        nphi_valid = np.isfinite(nphi)
        dtc_valid = np.isfinite(dtc)
        gr_valid = np.isfinite(gr)

        if nphi_valid and rhob < 1.95 and nphi > 0.35:
            return "coal"

        if (((v_anhydrite > 0.55) and (not nphi_valid or nphi < 0.10) and (not dtc_valid or dtc < 70))
                or (rhob > 2.90 and (not nphi_valid or nphi < 0.08) and (not gr_valid or gr < 60)
                    and (not dtc_valid or dtc < 70))
                or (rhob < 2.25 and (nphi_valid and nphi < 0.05) and (not gr_valid or gr < 35))):
            return "anhydrite" if (v_anhydrite > 0.55 or rhob > 2.90) else "salt"

        if vcl > 0.45 or (gr_valid and gr > 150 and (not nphi_valid or nphi > 0.32)):
            return "shale"

        carbonate_signature = (
            (gr_valid and gr < 45 and rhob > 2.38 and (not nphi_valid or nphi < 0.24) and (not dtc_valid or dtc < 90))
            or (carbonate_volume >= max(v_quartz, 0.20) and gr_valid and gr < 55 and (not nphi_valid or nphi < 0.28))
        )
        if carbonate_signature:
            return "dolomite" if v_dolomite > v_calcite else "limestone"

        clastic_signature = (
            v_quartz > 0.25
            or (gr_valid and gr >= 35)
            or (nphi_valid and nphi > 0.20)
            or (dtc_valid and dtc > 80)
            or rhob < 2.45
        )
        if clastic_signature:
            return "sandstone"

        return "mixed"

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None
    ) -> LithologyResult:
        """
        Calculate lithology fractions.

        Parameters
        ----------
        well : Well
            Well with RHOB, VCL/PHIE, and enough NPHI/PEF/DTC information
            for a reduced or full mineral solve.
        vcl : np.ndarray, optional
            Clay volume (uses VCL curve if not provided)

        Returns
        -------
        LithologyResult
            Lithology calculation results
        """
        n_samples = well.n_samples

        # Get required curves depth-wise so sparse primary aliases do not
        # block usable samples from lower-priority aliases.
        rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
        nphi, _, _ = get_curve_data_depthwise(well, ["NPHI", "PHIN"])
        pef, _, _ = get_curve_data_depthwise(well, ["PEF"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        dtc, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])
        gr, _, _ = get_curve_data_depthwise(well, ["GR"])
        # Core (lab) porosity, if present, pins the matrix density and lets the
        # solver resolve quartz/calcite even without PEF.
        core_phi, _, _ = get_curve_data_depthwise(
            well, ["PHIE_CC", "PHIT_CC", "CPOR", "CPOR_CC", "PHIC"]
        )

        if rhob is None:
            raise ValueError("RHOB with finite samples required for lithology")
        if nphi is None and pef is None and dtc is None:
            raise ValueError("NPHI/PHIN, PEF, or DTC with finite samples required for lithology")

        # Get VCL
        if vcl is None and self.include_clay:
            vcl_data, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
            vcl = vcl_data if vcl_data is not None else np.zeros(n_samples)
        elif vcl is None:
            vcl = np.zeros(n_samples)

        pef_clean = self._screen_pef(pef) if pef is not None else None
        dtc_clean = self._screen_dtc(dtc) if dtc is not None else None
        nphi_data = np.asarray(nphi, dtype=float).copy() if nphi is not None else np.full(n_samples, np.nan)
        phie_data = np.asarray(phie, dtype=float).copy() if phie is not None else np.zeros(n_samples)
        gr_data = np.asarray(gr, dtype=float).copy() if gr is not None else np.full(n_samples, np.nan)

        # Solve for minerals. The solver already removes clay and porosity
        # internally, so raw curves are passed here.
        solver, mineral_selection = self._select_solver(well)
        result = solver.solve(
            vcl=vcl,
            rhob=rhob,
            nphi=nphi,
            pef=pef_clean,
            phie=phie_data,
            dt=dtc_clean,
            core_phi=core_phi,
        )

        # A density-neutron suite cannot independently resolve quartz, calcite,
        # dolomite, anhydrite *and* porosity.  Where the full inversion is
        # underdetermined but an accepted workflow PHIE exists, use that PHIE as
        # a fixed constraint and solve the identifiable two-mineral
        # quartz-calcite system.  Keep dolomite/anhydrite unavailable rather than
        # manufacturing zero-valued curves.  Core porosity remains the preferred
        # constraint in the primary solve above.
        underdetermined = np.asarray(
            result.get("underdetermined", np.zeros(n_samples, dtype=bool)), dtype=bool
        )
        reduced_qc_fallback = np.zeros(n_samples, dtype=bool)
        if (
            type(self.solver) is MultiMineralSolver
            and not self._custom_minerals
            and mineral_selection["lithology"] != "carbonate"
            and phie is not None
            and np.any(underdetermined & np.isfinite(phie_data))
        ):
            reduced_solver = MultiMineralSolver(
                minerals=["quartz", "calcite"],
                minerals_no_pef=["quartz", "calcite"],
            )
            reduced = reduced_solver.solve(
                vcl=vcl,
                rhob=rhob,
                nphi=nphi,
                core_phi=phie_data,
            )
            reduced_solved = np.asarray(reduced["solved"], dtype=bool)
            reduced_qc_fallback = underdetermined & reduced_solved
            for mineral in ("quartz", "calcite"):
                target = np.asarray(result[mineral], dtype=float).copy()
                target[reduced_qc_fallback] = reduced[mineral][reduced_qc_fallback]
                result[mineral] = target
            for key in ("phie", "residual", "qc_flag", "solved"):
                target = np.asarray(result[key]).copy()
                target[reduced_qc_fallback] = reduced[key][reduced_qc_fallback]
                result[key] = target

        v_quartz = result.get("quartz", np.zeros(n_samples))
        v_calcite = result.get("calcite", np.zeros(n_samples))
        v_dolomite = result.get("dolomite", np.zeros(n_samples))
        v_anhydrite = result.get("anhydrite", np.zeros(n_samples))

        # Determine dominant lithology.
        # Primary path: classify each solved sample from its mineral volumes so the
        # lithology track and the mineral-volumes track share a single source of
        # truth. Fall back to the raw-log cutoff tree only where the solver could
        # not run (too few curves -> solved == False).
        solved = np.asarray(result.get("solved", np.zeros(n_samples, dtype=bool)), dtype=bool)
        lithology_code = np.zeros(n_samples, dtype=int)
        lithology_name = np.empty(n_samples, dtype=object)
        rhomaa = np.full(n_samples, np.nan)
        umaa = np.full(n_samples, np.nan)
        litho_disagree = np.zeros(n_samples, dtype=bool)

        # Vectorized volume-based classification for the solved samples (identical
        # to the per-sample _classify_from_volumes; avoids a Python loop over depth).
        vol_codes = self._classify_codes_from_volumes(
            vcl, v_quartz, v_calcite, v_dolomite, v_anhydrite, rhob, nphi_data, gr_data
        )
        code_to_name = {v: k for k, v in self.LITHOLOGY_CODES.items()}

        for i in range(n_samples):
            if solved[i]:
                lith_name = code_to_name.get(int(vol_codes[i]), "mixed")
            else:
                lith_name = self._classify_sample(
                    rhob=float(rhob[i]),
                    nphi=float(nphi_data[i]),
                    gr=float(gr_data[i]) if np.isfinite(gr_data[i]) else np.nan,
                    dtc=float(dtc_clean[i]) if dtc_clean is not None and np.isfinite(dtc_clean[i]) else np.nan,
                    vcl=float(vcl[i]),
                    v_quartz=float(v_quartz[i]),
                    v_calcite=float(v_calcite[i]),
                    v_dolomite=float(v_dolomite[i]),
                    v_anhydrite=float(v_anhydrite[i]),
                )
            lithology_name[i] = lith_name
            lithology_code[i] = self.LITHOLOGY_CODES.get(lith_name, self.LITHOLOGY_CODES["mixed"])

            # Automated RHOmaa-Umaa cross-check on clean quartz/calcite/dolomite
            # samples: an independent ternary solve that flags disagreement with the
            # volume-based call (no crossplot/human needed).
            if (pef_clean is not None and solved[i]
                    and lith_name in self._QCD_ENDPOINTS
                    and float(vcl[i]) < self.SHALE_CLAY_CUT):
                rm, um = self._rhomaa_umaa(
                    float(rhob[i]), float(nphi_data[i]), float(pef_clean[i])
                )
                rhomaa[i] = rm
                umaa[i] = um
                xlith = self._ternary_qcd_lithology(rm, um)
                if xlith is not None and xlith != lith_name:
                    litho_disagree[i] = True

        return LithologyResult(
            v_quartz=v_quartz,
            v_calcite=v_calcite,
            v_dolomite=v_dolomite,
            v_anhydrite=v_anhydrite,
            v_clay=vcl,
            lithology_code=lithology_code,
            lithology_name=lithology_name,
            qc_flag=np.asarray(result.get("qc_flag", np.zeros(n_samples, dtype=bool)), dtype=bool),
            solved=solved,
            reduced_qc_fallback=reduced_qc_fallback,
            rhomaa=rhomaa,
            umaa=umaa,
            litho_disagree=litho_disagree,
            mineral_selection=mineral_selection,
        )

    def add_to_well(
        self,
        well: "Well",
        result: Optional[LithologyResult] = None,
        include_lithology_code: bool = True,
    ) -> "Well":
        """
        Add lithology curves to well.

        Parameters
        ----------
        well : Well
            Well object
        result : LithologyResult, optional
            Pre-computed result
        include_lithology_code : bool, optional
            Write the fallback ``LITHO`` curve derived from mineral volumes.
            Set False when an interpreted input lithology curve is already
            present and should remain authoritative.

        Returns
        -------
        Well
            Well with lithology curves
        """
        if result is None:
            result = self.calculate(well)

        well.add_curve("VQUAR", result.v_quartz, "v/v", "Quartz Volume")
        well.add_curve("VCALC", result.v_calcite, "v/v", "Calcite Volume")
        well.add_curve("VDOLO", result.v_dolomite, "v/v", "Dolomite Volume")
        well.add_curve("VANHY", result.v_anhydrite, "v/v", "Anhydrite Volume")
        well.add_curve("MMQC", result.qc_flag.astype(float), "", "Multi-mineral QC flag (1 = poor fit or unsolved)")
        if include_lithology_code:
            well.add_curve("LITHO", result.lithology_code.astype(float), "", "Lithology Code")
            # The RHOmaa-Umaa cross-check pertains to the computed lithology, so it
            # is only emitted alongside it (not when a vendor curve is authoritative).
            if result.litho_disagree is not None and np.any(result.litho_disagree):
                well.add_curve(
                    "LITHO_XQC",
                    np.asarray(result.litho_disagree, dtype=float),
                    "",
                    "Lithology cross-check flag (1 = RHOmaa-Umaa disagrees with volume call)",
                )

        return well


def calculate_lithology(
    well: "Well",
    vcl: Optional[np.ndarray] = None,
    **kwargs
) -> LithologyResult:
    """
    Calculate lithology for a well.

    Convenience function for LithologyModule.

    Parameters
    ----------
    well : Well
        Well object
    vcl : np.ndarray, optional
        Clay volume
    **kwargs
        Additional arguments for LithologyModule

    Returns
    -------
    LithologyResult
        Lithology results

    Examples
    --------
    >>> from petrophys.interpretation.lithology import calculate_lithology
    >>> result = calculate_lithology(well)
    """
    module = LithologyModule(**kwargs)
    return module.calculate(well, vcl)
