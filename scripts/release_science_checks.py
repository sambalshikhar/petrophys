"""Independent numerical spot checks and contract probes for release evidence.

These checks are deliberately separate from the module implementations. A pass
verifies the stated case only, not an entire equation family or field validity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARCHIE = "https://personal.ems.psu.edu/~radovic/EME590_Archie_1942.pdf"
WYLLIE = "https://doi.org/10.1190/1.1438217"
NIST = "https://www.nist.gov/pml/us-surveyfoot/revised-unit-conversion-factors"


def run_checks():
    from petrophys.interpretation.saturation import archie
    from petrophys.interpretation.saturation.archie import archie_f, archie_ri
    from petrophys.interpretation.porosity import density_porosity, wyllie_time_average, effective_porosity
    from petrophys.interpretation.clay_volume.gr_method import vcl_linear
    from petrophys.interpretation.permeability import timur, coates_denoo, wyllie_rose
    from petrophys.interpretation.saturation.height import fit_pressure_fwl, leverett_j, pc_from_height
    from petrophys.model.units import UnitConverter
    from petrophys.interpretation.nmr.permeability import t2lm_from_bins, bvi_from_bins, ffi_from_bins
    from petrophys.interpretation.rock_physics.elastic import calculate_elastic_properties

    checks = []

    def check(key, symbol, purpose, function, expected, reference, tolerance=1e-10):
        started = time.monotonic()
        row = {"ID": key, "Callable": symbol, "Check": purpose, "Reference": reference,
               "Status": "PENDING", "Expected": str(expected), "Tolerance": tolerance,
               "Evidence": "outputs/release_audit/evidence/checks-independent-science.json",
               "Checked UTC": datetime.now(timezone.utc).isoformat()}
        try:
            actual = function()
            row["Actual"] = str(actual)
            if isinstance(expected, str) and expected == "ValueError":
                passed = False
            elif isinstance(expected, str) and expected == "NaN":
                passed = bool(np.isnan(actual).all())
            elif isinstance(expected, str) and expected == "same scalar/array":
                passed = bool(np.allclose(actual[0], actual[1], rtol=1e-12, atol=0.0, equal_nan=True))
            else:
                passed = bool(np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True))
            row["Status"] = "PASS" if passed else "FAIL"
        except ValueError as exc:
            row.update(Status="PASS" if expected == "ValueError" else "FAIL", Actual=f"ValueError: {exc}")
        except Exception as exc:
            row.update(Status="FAIL", Actual=f"{type(exc).__name__}: {exc}")
        row["Seconds"] = round(time.monotonic() - started, 6)
        checks.append(row)

    check("ARCHIE-001", "petrophys.interpretation.saturation.archie.archie", "Clean sandstone: Rt=10 ohm.m, Rw=0.05 ohm.m, phi=0.2 v/v; a=1,m=n=2",
          lambda: archie(10.0, .05, .2), math.sqrt(.125), ARCHIE)
    check("ARCHIE-002", "petrophys.interpretation.saturation.archie.archie_f", "Formation factor at phi=0.2 v/v, a=1,m=2",
          lambda: archie_f(.2), 25., ARCHIE)
    check("ARCHIE-003", "petrophys.interpretation.saturation.archie.archie_ri", "Resistivity index at Sw=0.5 v/v, n=2",
          lambda: archie_ri(.5), 4., ARCHIE)
    check("DENSITY-001", "petrophys.interpretation.porosity.density_porosity.density_porosity", "Mass balance: 80% quartz (2.65 g/cc), 20% water (1 g/cc) gives bulk density 2.32 g/cc",
          lambda: density_porosity(2.32), .2, "Independent two-component mass balance")
    check("WYLLIE-001", "petrophys.interpretation.porosity.sonic_porosity.wyllie_time_average", "Time mixing: 80% matrix 55.5 us/ft and 20% fluid 189 us/ft gives 82.2 us/ft",
          lambda: wyllie_time_average(82.2), .2, WYLLIE)
    check("WYLLIE-002", "petrophys.interpretation.porosity.sonic_porosity.wyllie_time_average", "Reversed endpoints: 80% matrix 230 us/ft plus 20% fluid 189 us/ft gives 221.8 us/ft",
          lambda: wyllie_time_average(221.8,230.,189.), .2, "Independent time/volume balance; the denominator can be negative")
    check("DENSITY-002", "petrophys.interpretation.porosity.density_porosity.density_porosity", "Reversed endpoints: 80% matrix 1.5 g/cc plus 20% fluid 3 g/cc gives 1.8 g/cc",
          lambda: density_porosity(1.8,1.5,3.), .2, "Independent mass/volume balance; the denominator can be negative")
    check("PHIE-001", "petrophys.interpretation.porosity.effective_porosity.effective_porosity", "0.25 total porosity minus 0.2 clay fraction times 0.15 clay porosity",
          lambda: effective_porosity(.25, .2, .15), .22, "Independent pore-volume balance; clay-bound volume=0.03 v/v")
    check("UNITS-001", "petrophys.model.units.UnitConverter.depth", "1 international foot -> 0.3048 m exactly",
          lambda: UnitConverter.depth(1., "ft", "m"), .3048, NIST)
    check("UNITS-002", "petrophys.model.units.UnitConverter.temperature", "32 degF -> 0 degC",
          lambda: UnitConverter.temperature(32., "F", "C"), 0., "Exact affine Fahrenheit/Celsius relation")
    check("UNITS-003", "petrophys.model.units.UnitConverter.temperature", "273.15 K -> 0 degC",
          lambda: UnitConverter.temperature(273.15, "K", "C"), 0., "Exact Kelvin/Celsius offset")
    check("UNITS-004", "petrophys.model.units.UnitConverter.porosity", "20 pu -> 0.2 v/v",
          lambda: UnitConverter.porosity(20., "pu", "v/v"), .2, "Percent is one hundredth")
    # 1 mD=9.869233e-16 m2, 1 psi=6894.757293168 Pa, 1 dyne/cm=0.001 N/m.
    j_si = (10. * 6894.757293168) * math.sqrt(100. * 9.869233e-16 / .2) / (.030 * math.cos(math.pi / 6))
    check("J-001", "petrophys.interpretation.saturation.height.capillary.leverett_j", "SI dimensional check: Pc=10 psi,k=100 mD,phi=0.2,sigma=30 dyne/cm,theta=30 deg",
          lambda: leverett_j(10.,100.,.2,30.,30.), j_si, "Leverett definition J=Pc*sqrt(k/phi)/(sigma*cos(theta)); SI comparison allows 0.1% for the documented field-unit coefficient 0.21645", tolerance=1e-3)
    check("PC-001", "petrophys.interpretation.saturation.height.capillary.pc_from_height", "Hydrostatic buoyancy: 100 ft, density difference 200 kg/m3, g=9.80665 m/s2",
          lambda: pc_from_height(100., 1., .8), 200.*9.80665*30.48/6894.757293168,
          "First-principles pressure=rho*g*h; 0.2% tolerance for documented approximate 0.433 psi/ft field coefficient", tolerance=2e-3)
    depths = np.array([1300., 1400., 1600., 1700.])
    pressures = np.array([187.,193.5,211.,222.])
    check("FWL-001", "petrophys.interpretation.saturation.height.formation_pressure.fit_pressure_fwl", "Two independently constructed straight pressure legs meet at 1500 m TVD,200 bar",
          lambda: fit_pressure_fwl(depths,pressures,['oil','oil','water','water']).fwl_depth, 1500., "Analytic intersection of p_o=200+.065*(z-1500), p_w=200+.11*(z-1500)")
    check("FWL-002", "petrophys.interpretation.saturation.height.formation_pressure.fit_pressure_fwl", "FWL invariant under m/bar -> ft/psi conversion",
          lambda: fit_pressure_fwl(depths/.3048,pressures/.06894757293168,['oil','oil','water','water'], depth_unit='ft',pressure_unit='psi').fwl_depth*.3048,1500., "Analytic intersection plus independent exact depth conversion")
    for name, fn in (("timur",timur),("coates_denoo",coates_denoo),("wyllie_rose",wyllie_rose)):
        symbol = "petrophys.interpretation.permeability." + ("timur." if name=="timur" else "coates_denoo.") + name
        check("SHAPE-"+name, symbol, "Positive inputs phi=0.0005, Swi=0.005 must give the same result as scalars and single-element arrays",
              lambda fn=fn: (fn(.0005,.005),fn(np.array([.0005]),np.array([.005]))), "same scalar/array", "Public scalar/array contract, independent of equation coefficients")
    check("MISSING-DENSITY", "petrophys.interpretation.porosity.density_porosity.density_porosity", "Missing bulk density with coincident endpoints must not become zero porosity",
          lambda: density_porosity(float('nan'),1.,1.), "NaN", "Unknown measurement cannot determine porosity")
    check("MISSING-WYLLIE", "petrophys.interpretation.porosity.sonic_porosity.wyllie_time_average", "Missing sonic with coincident endpoints must not become zero porosity",
          lambda: wyllie_time_average(float('nan'),55.5,55.5), "NaN", "Unknown measurement cannot determine porosity")
    check("INVALID-ARCHIE", "petrophys.interpretation.saturation.archie.archie", "Infinite Rt is invalid input and must not become measured zero Sw",
          lambda: archie(float('inf'),.05,.2), "NaN", "Nonfinite measurement preserves unknown status")
    check("MISSING-GR", "petrophys.interpretation.clay_volume.gr_method.vcl_linear", "Missing GR with coincident endpoints must not become clean rock",
          lambda: vcl_linear(np.array([float('nan')]),20.,20.), "NaN", "Unknown GR cannot establish a clean formation")
    check("NMR-LM-001", "petrophys.interpretation.nmr.permeability.t2lm_from_bins",
          "Equal pore volumes at 10 and 40 ms have geometric mean 20 ms",
          lambda: t2lm_from_bins([[2., 2.]], [10., 40.]), [20.], "Independent equal-weight geometric mean sqrt(10*40)")
    check("NMR-LM-MISSING", "petrophys.interpretation.nmr.permeability.t2lm_from_bins",
          "An incomplete distribution cannot determine a complete logarithmic mean",
          lambda: t2lm_from_bins([[2., float('nan')]], [10., 40.]), "NaN", "Default complete-bin coverage requirement")
    check("NMR-PARTIAL-BVI", "petrophys.interpretation.nmr.permeability.bvi_from_bins",
          "Cutoff through the log-bin centre splits its volume in half: 1+2/2=2 pu",
          lambda: bvi_from_bins([[1., 2., 3.]], [4., 16., 64.], 16.), [2.], "Symmetric log-spaced bins; half of the central bin is below its centre")
    check("NMR-PARTIAL-FFI", "petrophys.interpretation.nmr.permeability.ffi_from_bins",
          "Complementary free-fluid volume: 2/2+3=4 pu",
          lambda: ffi_from_bins([[1., 2., 3.]], [4., 16., 64.], 16.), [4.], "Independent bin-volume balance; BVI+FFI=6 pu")
    # Construct a homogeneous isotropic elastic medium: rho=2500 kg/m3,
    # G=10 GPa, K=80/3 GPa. Its wave speeds are Vs=2000, Vp=4000 m/s.
    def elastic():
        return calculate_elastic_properties(2.5, 76.2, 152.4)
    for field, expected, definition in (
        ("vp_m_s", 4000., "SI wave speed from travel time 76.2 us per 0.3048 m"),
        ("shear_mod_gpa", 10., "Shear wave speed squared equals G/rho"),
        ("bulk_mod_gpa", 80./3., "Compressional wave speed squared equals (K+4G/3)/rho"),
        ("young_mod_gpa", 80./3., "Isotropic E=9KG/(3K+G), with K=80/3 and G=10 GPa"),
        ("poisson_ratio", 1./3., "Isotropic nu=(3K-2G)/(6K+2G)"),
        ("acoustic_impedance_mrayl", 10., "SI impedance rho*Vp=10 million kg/(m2 s)"),
    ):
        check("ELASTIC-" + field.upper(), "petrophys.interpretation.rock_physics.elastic.calculate_elastic_properties",
              definition, lambda field=field: getattr(elastic(), field), expected,
              "Independently constructed homogeneous isotropic medium; no static/dynamic equivalence assumed")
    check("ELASTIC-MISSING-SHEAR", "petrophys.interpretation.rock_physics.elastic.calculate_elastic_properties",
          "Missing shear sonic leaves shear-dependent moduli unknown",
          lambda: calculate_elastic_properties(2.5, 76.2).young_mod_gpa, "NaN",
          "Vp and density alone do not determine both independent elastic constants")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/release_audit/evidence/checks-independent-science.json")
    args = parser.parse_args()
    rows = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf8")
    for row in rows:
        print(row["Status"],row["ID"],row["Actual"])
    failed = sum(row["Status"] != "PASS" for row in rows)
    print(f"{len(rows)-failed}/{len(rows)} independent cases passed; failures are release blockers.")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
