"""Calcite and dolomite volumes of a carbonate from bulk density and closure.

With the clay volume and porosity fixed, a clay-calcite-dolomite rock leaves
two unknown bulk volumes and two equations::

    Vcal + Vdol                 = 1 - Vclay - PHI
    Vcal*rho_cal + Vdol*rho_dol = RHOB - PHI*rho_fl - Vclay*rho_clay

so the dolomite share of the carbonate matrix is
``(rho_maa - rho_cal) / (rho_dol - rho_cal)``, with the apparent matrix density
``rho_maa = (RHOB - PHI*rho_fl - Vclay*rho_clay) / (1 - PHI - Vclay)``
(Clavier and Rust, 1976, The Log Analyst 17(6)). The system is solved in closed
form for whole arrays at once. The dolomite volume is bounded to
``[0, 1 - Vclay - PHI]``, which is the bounded least-squares solution of the
density equation under closure, and a sample is solved where that bounded
solution reproduces RHOB within ``density_tolerance``. Elsewhere the apparent
matrix density lies outside calcite-dolomite (quartz, anhydrite, bad hole), so
the volumes are NaN and ``qc_flag`` is set.

The split is only as independent as the porosity. A porosity computed from
RHOB with an assumed matrix density gives that matrix back wherever it is
density-derived, so with the workflow's own PHIE or PHIT the split largely
restates ``matrix_type``; the MID plot takes its porosity from the
density-neutron crossplot instead. Where PEF is logged, the volumetric
photoelectric factor U (calcite 13.8, dolomite 9.0 b/cm3) separates the two
almost independently of porosity; this solve does not use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys.interpretation.lithology import costa_carbonate as costa
from petrophys.interpretation.lithology.lithology_module import LithologyResult
from petrophys.model.constants import MatrixProperties

if TYPE_CHECKING:
    from petrophys.model.well import Well


#: Largest RHOB misfit (g/cm3) of a solved sample: the nominal density-tool
#: uncertainty, as in the version 1 mineral solver's row weights.
DENSITY_TOLERANCE = 0.02

#: Constants of each calibration, densities in g/cm3. ``"generic"`` takes the
#: clay volume from VCL with a shale density, effective porosity, and calcite
#: and dolomite from the shared mineral table (Schlumberger Log Interpretation
#: Charts); the workflow passes its own ``rho_fl`` and ``rho_cl``. ``"costa"``
#: is the COSTA field calibration of the published notebooks: illite from GR,
#: total porosity, a 2.847 dolomite and a 1.1 brine.
CARBONATE_CALIBRATIONS: Dict[str, Dict[str, object]] = {
    "generic": {
        "clay_source": "VCL",
        "porosity_curves": ("PHIE", "PHIT"),
        "rho_fluid": 1.0,
        "rho_clay": 2.45,
        "rho_calcite": MatrixProperties.get("calcite").density,
        "rho_dolomite": MatrixProperties.get("dolomite").density,
    },
    "costa": {
        "clay_source": f"VOL_ILLITE = {costa.ILLITE_GR_INTERCEPT} + {costa.ILLITE_GR_SLOPE} * GR",
        "porosity_curves": ("PHIT", "PHIE"),
        "rho_fluid": costa.COSTA_FLUID_DENSITY,
        "rho_clay": costa.ILLITE_DENSITY,
        "rho_calcite": costa.CALCITE_DENSITY,
        "rho_dolomite": costa.DOLOMITE_DENSITY,
    },
}

_LITHOLOGY_CODES = {"shale": 4, "limestone": 2, "dolomite": 3}


def solve_calcite_dolomite(
    rhob,
    porosity,
    clay,
    *,
    rho_fluid: float,
    rho_clay: float,
    rho_calcite: float = 2.71,
    rho_dolomite: float = 2.87,
    density_tolerance: float = DENSITY_TOLERANCE,
) -> Dict[str, np.ndarray]:
    """Bulk calcite and dolomite volumes from bulk density and closure.

    Parameters
    ----------
    rhob, porosity, clay : array_like
        Bulk density (g/cm3), porosity and clay bulk volume (v/v).
    rho_fluid, rho_clay, rho_calcite, rho_dolomite : float
        Pore-fluid, clay, calcite and dolomite densities, g/cm3.
    density_tolerance : float, optional
        Largest RHOB misfit, g/cm3, of a solved sample (default 0.02).

    Returns
    -------
    dict
        ``calcite`` and ``dolomite`` bulk volumes (NaN where not solved),
        ``density_misfit`` (modelled minus measured RHOB of the bounded
        solution, g/cm3) and ``solved``.

    Examples
    --------
    >>> out = solve_calcite_dolomite(2.432, 0.2, 0.0, rho_fluid=1.0, rho_clay=2.45)
    >>> round(float(out["dolomite"]), 3), round(float(out["calcite"]), 3)
    (0.4, 0.4)
    """
    rhob, porosity, clay = np.broadcast_arrays(
        *(np.asarray(values, dtype=float) for values in (rhob, porosity, clay))
    )
    matrix = 1.0 - clay - porosity
    # RHOB left over once the matrix is taken as calcite: dolomite explains it.
    excess = rhob - porosity * rho_fluid - clay * rho_clay - matrix * rho_calcite
    contrast = rho_dolomite - rho_calcite
    with np.errstate(invalid="ignore"):
        dolomite = np.clip(excess / contrast, 0.0, np.maximum(matrix, 0.0))
    misfit = contrast * dolomite - excess
    solved = np.isfinite(misfit) & (matrix >= 0.0) & (np.abs(misfit) <= density_tolerance)
    dolomite = np.where(solved, dolomite, np.nan)
    return {
        "calcite": np.where(solved, matrix - dolomite, np.nan),
        "dolomite": dolomite,
        "density_misfit": misfit,
        "solved": solved,
    }


def solve_carbonate_minerals(
    well: "Well",
    calibration: str = "generic",
    *,
    rho_fluid: Optional[float] = None,
    rho_clay: Optional[float] = None,
    density_tolerance: float = DENSITY_TOLERANCE,
) -> LithologyResult:
    """Clay, calcite and dolomite bulk volumes of a carbonate well.

    Parameters
    ----------
    well : Well
        Needs RHOB, a porosity curve and VCL (``"generic"``) or GR
        (``"costa"``).
    calibration : {"generic", "costa"}, optional
        ``"generic"`` (default): clay volume from VCL, effective porosity
        (PHIE, else PHIT), fluid 1.0 and clay 2.45 g/cm3 unless given, and
        calcite 2.71 and dolomite 2.87 g/cm3. ``"costa"``: the COSTA field
        calibration, illite volume ``-0.0424035 + 0.00562239 * GR`` (not
        clipped), total porosity (PHIT, else PHIE), illite 2.52, calcite
        2.71, dolomite 2.847 and brine 1.1 g/cm3.
    rho_fluid, rho_clay : float, optional
        Pore-fluid and clay densities, g/cm3, replacing the calibration's.
    density_tolerance : float, optional
        Largest RHOB misfit, g/cm3, of a solved sample (default 0.02).

    Returns
    -------
    LithologyResult
        Bulk volumes (NaN where not solved; quartz and anhydrite are 0 where
        solved), the clay volume used as ``v_clay``, the dominant component as
        lithology (``unknown`` where not solved) and ``mineral_selection``
        recording the calibration and constants.

    Raises
    ------
    ValueError
        For an unknown calibration or missing input curves.
    """
    if calibration not in CARBONATE_CALIBRATIONS:
        raise ValueError(
            f"calibration must be one of {', '.join(CARBONATE_CALIBRATIONS)}; got {calibration!r}"
        )
    constants = CARBONATE_CALIBRATIONS[calibration]
    rho_fluid = float(constants["rho_fluid"] if rho_fluid is None else rho_fluid)
    rho_clay = float(constants["rho_clay"] if rho_clay is None else rho_clay)

    rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
    porosity, porosity_source, _ = get_curve_data_depthwise(well, list(constants["porosity_curves"]))
    clay_source = constants["clay_source"]
    if calibration == "costa":
        gr, _, _ = get_curve_data_depthwise(well, ["GR", "CGR", "SGR"])
        clay = None if gr is None else costa.ILLITE_GR_INTERCEPT + costa.ILLITE_GR_SLOPE * gr
        needs = "RHOB, GR and PHIT/PHIE"
    else:
        clay, clay_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        needs = "RHOB, VCL and PHIE/PHIT"
    if rhob is None or porosity is None or clay is None:
        raise ValueError(f"The {calibration} carbonate inversion requires {needs}")

    solution = solve_calcite_dolomite(
        rhob, porosity, clay,
        rho_fluid=rho_fluid,
        rho_clay=rho_clay,
        rho_calcite=float(constants["rho_calcite"]),
        rho_dolomite=float(constants["rho_dolomite"]),
        density_tolerance=density_tolerance,
    )
    solved = solution["solved"]
    clay = np.asarray(clay, dtype=float)
    stack = np.column_stack((clay, solution["calcite"], solution["dolomite"]))
    dominant = np.argmax(np.where(np.isfinite(stack), stack, -np.inf), axis=1)
    names = np.full(well.n_samples, "unknown", dtype=object)
    for index, name in enumerate(_LITHOLOGY_CODES):
        names[solved & (dominant == index)] = name
    codes = np.zeros(well.n_samples, dtype=int)
    for name, code in _LITHOLOGY_CODES.items():
        codes[names == name] = code
    zero_where_solved = np.where(solved, 0.0, np.nan)

    return LithologyResult(
        v_quartz=zero_where_solved,
        v_calcite=solution["calcite"],
        v_dolomite=solution["dolomite"],
        v_anhydrite=zero_where_solved.copy(),
        v_clay=clay,
        lithology_code=codes,
        lithology_name=names,
        qc_flag=~solved,
        solved=solved,
        reduced_qc_fallback=np.zeros(well.n_samples, dtype=bool),
        mineral_selection={
            "lithology_domain": "carbonate",
            "method": f"{calibration}_calcite_dolomite_density_closure",
            "carbonate_calibration": calibration,
            "clay_source": clay_source,
            "porosity_curve": porosity_source,
            "fluid_density": rho_fluid,
            "clay_density": rho_clay,
            "calcite_density": float(constants["rho_calcite"]),
            "dolomite_density": float(constants["rho_dolomite"]),
            "density_tolerance": float(density_tolerance),
        },
    )
