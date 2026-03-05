"""
Demo 02: Granular Module Usage with Real LAS Data
===================================================

This demo shows how to use individual petrophys modules independently:
- Clay Volume (VCL) from GR, SP, and N-D methods
- Clay Resistivity (Rcl)
- Porosity (density, sonic, total, effective, secondary)
- Water Saturation with model routing
- Permeability with impermeability detection
- Rock Quality (RQI, FZI)
- Net Pay with lithology-aware cutoffs

Well: 30/6-5 (Norwegian Continental Shelf)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import read_las
from petrophys.qaqc.standardization import standardize_curves

# ─────────────────────────────────────────────
# Load and standardize
# ─────────────────────────────────────────────
LAS_PATH = os.path.expanduser(
    "~/energy_gpt/EvalGen2/geolink_las/30_6-5.las"
)

print("=" * 60)
print("DEMO 02: Granular Module Usage - Well 30/6-5")
print("=" * 60)

well = read_las(LAS_PATH)
well = standardize_curves(well, rename_curves=True, convert_units=True)
print(f"Loaded well: {well.header.well_name}")
print(f"Available curves: {well.curves}")

# ─────────────────────────────────────────────
# 1. Clay Volume (VCL)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. CLAY VOLUME (VCL)")
print("=" * 60)

from petrophys.formation.clay_volume import calculate_vcl, vcl_from_gr, calculate_rcl

gr_curve = well.get("GR")
if gr_curve is not None:
    gr_data = gr_curve.data
    valid_gr = gr_data[~np.isnan(gr_data)]

    # Auto-detect GR range from percentiles
    gr_min = float(np.percentile(valid_gr, 5))
    gr_max = float(np.percentile(valid_gr, 95))
    print(f"GR range: P5={gr_min:.1f}, P95={gr_max:.1f} gAPI")

    # Method 1: Direct GR-based VCL (linear)
    vcl_linear = vcl_from_gr(gr_data, gr_min=gr_min, gr_max=gr_max, method="linear")
    print(f"VCL (linear):    mean={np.nanmean(vcl_linear):.3f}, "
          f"min={np.nanmin(vcl_linear):.3f}, max={np.nanmax(vcl_linear):.3f}")

    # Method 2: Larionov (tertiary rocks)
    vcl_larionov = vcl_from_gr(gr_data, gr_min=gr_min, gr_max=gr_max, method="larionov")
    print(f"VCL (Larionov):  mean={np.nanmean(vcl_larionov):.3f}")

    # Method 3: Multi-method VCL via module (uses GR + SP + N-D, takes minimum)
    result = calculate_vcl(well, gr_min=gr_min, gr_max=gr_max)
    well.add_curve("VCL", result.vcl, "v/v", "Clay Volume")
    print(f"VCL (composite): mean={np.nanmean(result.vcl):.3f}")

# ─────────────────────────────────────────────
# 2. Clay Resistivity (Rcl)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLAY RESISTIVITY (Rcl)")
print("=" * 60)

rdep_curve = well.get("RDEP")
vcl_curve = well.get("VCL")

if rdep_curve is not None and vcl_curve is not None:
    rcl = calculate_rcl(rdep_curve.data, vcl_curve.data, vcl_threshold=0.9)
    print(f"Rcl (from pure shale zones, VCL > 0.9): {rcl:.3f} ohm.m")
else:
    rcl = 5.0
    print(f"Using default Rcl: {rcl:.3f} ohm.m")

# ─────────────────────────────────────────────
# 3. Porosity
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. POROSITY")
print("=" * 60)

from petrophys.petrophysics.porosity import calculate_porosity

por_result = calculate_porosity(well, matrix="sandstone")

print(f"Density porosity (PHID):    mean={np.nanmean(por_result.phid):.3f}")
print(f"Total porosity (PHIT):      mean={np.nanmean(por_result.phit):.3f}")
print(f"Effective porosity (PHIE):  mean={np.nanmean(por_result.phie):.3f}")
if por_result.phis is not None:
    print(f"Sonic porosity (PHIS):      mean={np.nanmean(por_result.phis):.3f}")
if por_result.phi_sec is not None:
    print(f"Secondary porosity:         mean={np.nanmean(por_result.phi_sec):.3f}")

# Add porosity curves to well
well.add_curve("PHIT", por_result.phit, "v/v", "Total Porosity")
well.add_curve("PHIE", por_result.phie, "v/v", "Effective Porosity")
well.add_curve("PHID", por_result.phid, "v/v", "Density Porosity")
if por_result.phis is not None:
    well.add_curve("PHIS", por_result.phis, "v/v", "Sonic Porosity")

# ─────────────────────────────────────────────
# 4. Water Saturation (Sw) - Model Routing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. WATER SATURATION (Sw)")
print("=" * 60)

from petrophys.petrophysics.saturation import (
    calculate_sw, archie, simandoux, indonesia
)

# a) Direct Archie on a sample point (clean zone, VCL < 0.3)
sw_point = archie(rt=10.0, rw=0.05, phi=0.15, a=1.0, m=2.0, n=2.0)
print(f"Archie example (Rt=10, phi=0.15): Sw = {sw_point:.3f}")

# b) Auto-routed Sw calculation using well data
#    For clastic with salinity >= 20kppm, shaly zones route to Simandoux
rw = 0.05  # formation water resistivity
salinity_ppm = 35000  # North Sea typical

sw_result = calculate_sw(
    well,
    rw=rw,
    lithology="clastic",
    salinity=salinity_ppm,
    rcl=rcl,
)

well.add_curve("SW", sw_result.sw, "v/v", f"Water Saturation ({sw_result.model})")
print(f"Sw model used: {sw_result.model}")
print(f"Sw: mean={np.nanmean(sw_result.sw):.3f}, "
      f"min={np.nanmin(sw_result.sw):.3f}, max={np.nanmax(sw_result.sw):.3f}")

if sw_result.sxo is not None:
    well.add_curve("SXO", sw_result.sxo, "v/v", "Flushed Zone Sw")
    print(f"Sxo: mean={np.nanmean(sw_result.sxo):.3f}")

# ─────────────────────────────────────────────
# 5. Permeability + Impermeability Detection
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PERMEABILITY")
print("=" * 60)

from petrophys.petrophysics.permeability import calculate_permeability, flag_impermeable

perm_result = calculate_permeability(well, method="timur")
well.add_curve("PERM", perm_result.perm, "mD", f"Permeability ({perm_result.method})")

valid_perm = perm_result.perm[~np.isnan(perm_result.perm)]
valid_perm = valid_perm[valid_perm > 0]
if len(valid_perm) > 0:
    print(f"Permeability: P10={np.percentile(valid_perm, 10):.2f} mD, "
          f"median={np.median(valid_perm):.2f} mD, "
          f"P90={np.percentile(valid_perm, 90):.2f} mD")

# Impermeability flag (where sonic porosity >> density porosity)
phis = well.get("PHIS")
phid = well.get("PHID")
nphi = well.get("NPHI")
if phis is not None and phid is not None:
    phi_n = nphi.data if nphi is not None else phid.data
    imperm = flag_impermeable(phis.data, phid.data, phi_n)
    well.add_curve("IMPERM_FLAG", imperm.astype(float), "", "Impermeable Flag")
    print(f"Impermeable zones: {np.nansum(imperm):.0f} / {len(imperm)} samples "
          f"({100.0 * np.nansum(imperm) / len(imperm):.1f}%)")

# ─────────────────────────────────────────────
# 6. Rock Quality (RQI, FZI)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ROCK QUALITY (RQI, FZI)")
print("=" * 60)

from petrophys.reservoir import calculate_rock_quality, calculate_rqi, calculate_fzi

# Scalar example
rq = calculate_rock_quality(perm_k=100.0, phi_e=0.20)
print(f"Example (k=100 mD, phi_e=0.20):")
print(f"  RQI  = {rq['RQI']:.4f}")
print(f"  phi_z = {rq['phi_z']:.4f}")
print(f"  FZI  = {rq['FZI']:.4f}")

# Array-based from well
phie_data = well.get("PHIE").data
perm_data = well.get("PERM").data
mask = (~np.isnan(phie_data) & ~np.isnan(perm_data) &
        (phie_data > 0.01) & (perm_data > 0))
rqi_arr = calculate_rqi(perm_data[mask], phie_data[mask])
print(f"\nWell-derived RQI: median={np.median(rqi_arr):.4f}, "
      f"P90={np.percentile(rqi_arr, 90):.4f}")

# ─────────────────────────────────────────────
# 7. Net Pay (lithology-aware cutoffs)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. NET PAY")
print("=" * 60)

from petrophys.reservoir import calculate_net_pay

# Clastic cutoffs: VCL <= 0.3, PHIE >= 0.08, Sw <= 0.7
np_result = calculate_net_pay(well, lithology="clastic")
depth = well._depth

net_rock_count = np.nansum(np_result.net_rock)
net_res_count = np.nansum(np_result.net_reservoir)
net_pay_count = np.nansum(np_result.net_pay)
total = len(np_result.net_pay)

print(f"Lithology: clastic (PHIE cutoff = 0.08)")
print(f"Net Rock      : {net_rock_count:.0f} / {total} samples ({100.0 * net_rock_count / total:.1f}%)")
print(f"Net Reservoir : {net_res_count:.0f} / {total} samples ({100.0 * net_res_count / total:.1f}%)")
print(f"Net Pay       : {net_pay_count:.0f} / {total} samples ({100.0 * net_pay_count / total:.1f}%)")

# Also try carbonate cutoffs for comparison
np_result_carb = calculate_net_pay(well, lithology="carbonate")
net_pay_carb = np.nansum(np_result_carb.net_pay)
print(f"\nCarbonate cutoff (PHIE >= 0.045): Net Pay = {net_pay_carb:.0f} samples")

print("\nDone!")
