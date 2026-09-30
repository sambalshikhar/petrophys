"""Carbonate mineral inversion used by the published COSTA notebooks."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys.interpretation.lithology.lithology_module import LithologyResult


ILLITE_GR_INTERCEPT = -0.0424035
ILLITE_GR_SLOPE = 0.00562239
ILLITE_DENSITY = 2.52
CALCITE_DENSITY = 2.71
DOLOMITE_DENSITY = 2.847
COSTA_FLUID_DENSITY = 1.19


def solve_costa_carbonate(well) -> LithologyResult:
    """Solve COSTA illite, calcite and dolomite bulk volumes depth by depth.

    The implementation intentionally follows the source notebooks: bulk
    illite is a linear function of GR, the solid illite fraction is normalized
    by ``1 - PHIT``, and bounded SLSQP solves the density and closure
    constraints. The historical notebooks did not clip the GR-derived illite
    value before solving, so this function records infeasible rows through
    ``qc_flag`` rather than silently changing the equation.
    """
    rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
    gr, _, _ = get_curve_data_depthwise(well, ["GR", "CGR", "SGR"])
    phit, _, _ = get_curve_data_depthwise(well, ["PHIT", "PHIE"])
    if rhob is None or gr is None or phit is None:
        raise ValueError("COSTA carbonate inversion requires RHOB, GR and PHIT/PHIE")

    rhob = np.asarray(rhob, dtype=float)
    gr = np.asarray(gr, dtype=float)
    phit = np.asarray(phit, dtype=float)
    n = well.n_samples
    illite_bulk = ILLITE_GR_INTERCEPT + ILLITE_GR_SLOPE * gr
    calcite_bulk = np.full(n, np.nan, dtype=float)
    dolomite_bulk = np.full(n, np.nan, dtype=float)
    solved = np.zeros(n, dtype=bool)

    for index in range(n):
        if not np.all(np.isfinite([rhob[index], gr[index], phit[index]])):
            continue
        matrix_fraction = 1.0 - phit[index]
        if matrix_fraction <= 0.0:
            continue
        illite_solid = illite_bulk[index] / matrix_fraction

        def objective(values):
            return illite_solid + values[0] + values[1] - 1.0

        constraints = (
            {
                "type": "eq",
                "fun": lambda values: (
                    (
                        ILLITE_DENSITY * illite_solid
                        + CALCITE_DENSITY * values[0]
                        + DOLOMITE_DENSITY * values[1]
                    )
                    * matrix_fraction
                    + COSTA_FLUID_DENSITY * phit[index]
                    - rhob[index]
                ),
            },
            {
                "type": "eq",
                "fun": lambda values: illite_solid + values[0] + values[1] - 1.0,
            },
        )
        result = optimize.minimize(
            objective,
            [0.5, 0.5],
            bounds=((0.0, 1.0), (0.0, 1.0)),
            constraints=constraints,
            options={"disp": False},
        )
        calcite_bulk[index] = float(result.x[0]) * matrix_fraction
        dolomite_bulk[index] = float(result.x[1]) * matrix_fraction
        solved[index] = bool(result.success) and abs(float(objective(result.x))) <= 1e-5

    quartz = np.zeros(n, dtype=float)
    anhydrite = np.zeros(n, dtype=float)
    codes = np.zeros(n, dtype=int)
    names = np.full(n, "unknown", dtype=object)
    finite = np.isfinite(illite_bulk) & np.isfinite(calcite_bulk) & np.isfinite(dolomite_bulk)
    mineral_stack = np.column_stack((illite_bulk, calcite_bulk, dolomite_bulk))
    dominant = np.argmax(np.where(np.isfinite(mineral_stack), mineral_stack, -np.inf), axis=1)
    names[finite & (dominant == 0)] = "shale"
    names[finite & (dominant == 1)] = "limestone"
    names[finite & (dominant == 2)] = "dolomite"
    code_map = {"shale": 4, "limestone": 2, "dolomite": 3}
    for name, code in code_map.items():
        codes[names == name] = code

    return LithologyResult(
        v_quartz=quartz,
        v_calcite=calcite_bulk,
        v_dolomite=dolomite_bulk,
        v_anhydrite=anhydrite,
        v_clay=illite_bulk,
        lithology_code=codes,
        lithology_name=names,
        qc_flag=~solved,
        solved=solved,
        reduced_qc_fallback=np.zeros(n, dtype=bool),
        mineral_selection={
            "lithology_domain": "carbonate",
            "method": "costa_gr_illite_calcite_dolomite",
            "illite_equation": "-0.0424035 + 0.00562239 * GR",
            "fluid_density": COSTA_FLUID_DENSITY,
        },
    )
