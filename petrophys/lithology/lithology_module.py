"""
Main lithology determination module.

This module provides the LithologyModule class for comprehensive
lithology analysis from well logs.
"""
from typing import Optional, List
from dataclasses import dataclass
import numpy as np

from petrophys.lithology.multi_mineral_solver import (
    MultiMineralSolver,
)
from petrophys.utils.analysis_helpers import get_curve_data_depthwise


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
        include_clay: bool = True
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite", "anhydrite"]

        self.minerals = minerals
        self.include_clay = include_clay
        self.solver = MultiMineralSolver(minerals=minerals)

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
        result = self.solver.solve(
            vcl=vcl,
            rhob=rhob,
            nphi=nphi,
            pef=pef_clean,
            phie=phie_data,
            dt=dtc_clean,
        )

        v_quartz = result.get("quartz", np.zeros(n_samples))
        v_calcite = result.get("calcite", np.zeros(n_samples))
        v_dolomite = result.get("dolomite", np.zeros(n_samples))
        v_anhydrite = result.get("anhydrite", np.zeros(n_samples))

        # Determine dominant lithology
        lithology_code = np.zeros(n_samples, dtype=int)
        lithology_name = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
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

        return LithologyResult(
            v_quartz=v_quartz,
            v_calcite=v_calcite,
            v_dolomite=v_dolomite,
            v_anhydrite=v_anhydrite,
            v_clay=vcl,
            lithology_code=lithology_code,
            lithology_name=lithology_name,
            qc_flag=np.asarray(result.get("qc_flag", np.zeros(n_samples, dtype=bool)), dtype=bool),
            solved=np.asarray(result.get("solved", np.zeros(n_samples, dtype=bool)), dtype=bool),
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
    >>> from petrophys.lithology import calculate_lithology
    >>> result = calculate_lithology(well)
    """
    module = LithologyModule(**kwargs)
    return module.calculate(well, vcl)
