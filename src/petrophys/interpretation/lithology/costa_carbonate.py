"""COSTA field calibration of the carbonate mineral solve.

Craig Phillips's COSTA notebooks take bulk illite from GR, calcite and dolomite
from bulk density and closure, with the constants below.
:func:`~petrophys.interpretation.lithology.carbonate_minerals.solve_carbonate_minerals`
with ``calibration="costa"`` solves those equations in closed form for a whole
well; it reproduces the notebooks' per-sample SLSQP solve to 1e-7 v/v where
that converged, and also solves the samples it left unconverged.
"""

#: Bulk illite volume from GR: ``ILLITE_GR_INTERCEPT + ILLITE_GR_SLOPE * GR``.
ILLITE_GR_INTERCEPT = -0.0424035
ILLITE_GR_SLOPE = 0.00562239
#: Densities, g/cm3.
ILLITE_DENSITY = 2.52
CALCITE_DENSITY = 2.71
DOLOMITE_DENSITY = 2.847
#: The pore fluid of the notebooks' calcite-dolomite solve (FD = 1.1, set in the
#: neutron-density KNN section that runs before it).
COSTA_FLUID_DENSITY = 1.1
