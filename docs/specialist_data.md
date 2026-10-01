---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Specialist data

NMR, spectral gamma, core, saturation-height and borehole-image data each have their own API; their products join the same `Well`.
`pp.interpret` does not run spectral gamma or elemental spectroscopy; call those functions yourself.

```{code-cell} python
:tags: [remove-cell]

%matplotlib inline
import warnings

import matplotlib.pyplot as plt
import pandas as pd

# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The page explains this once, in the text.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
pd.set_option("display.max_colwidth", None)
plt.rcParams.update({"font.size": 9})
```

## NMR: Daydream CMR

The bundled Daydream 1 file is a vendor CMR summary, with GR and calipers.
{func}`~petrophys.visualization.create_nmr_panel` draws the standard display from its curves.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petrophys as pp
from petrophys.interpretation.nmr import NMRData
from petrophys.visualization import create_nmr_panel

cmr = pp.read(pp.sample("daydream1_cmr"))
create_nmr_panel(cmr, depth_range=(3900, 4100), figsize=(12, 8.5))
plt.show()
```

{class}`~petrophys.interpretation.nmr.NMRData` loads the same file to compute bound fluid (`MPHI - MFFI`), irreducible saturation and permeability.

```{code-cell} python
nmr = NMRData.from_las(pp.sample("daydream1_cmr"),
                       mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
print(f"{nmr.n_samples} samples, {nmr.top:.0f}-{nmr.bottom:.0f} m, "
      f"T2 cutoff {nmr.t2_cutoff:.0f} ms")
print("T2 bins:", nmr.has_bins, "| stored T2 log-mean:", nmr.has_t2lm)
print(f"Median MPHI {np.nanmedian(nmr.mphi):.1f} pu, "
      f"MBVI {np.nanmedian(nmr.mbvi):.1f} pu, Swirr {np.nanmedian(nmr.swirr_nmr()):.2f}")

k_library = nmr.permeability_coates(prefer_stored=False)  # textbook constants
k_vendor = nmr.stored_permeability["KTIM"]                 # the vendor's curve
both = (k_library > 0) & (k_vendor > 0)
bias = np.log10(k_library[both] / k_vendor[both])
print(f"Library vs vendor Timur-Coates: {both.sum()} samples, "
      f"mean {bias.mean():+.3f} decades, spread {bias.std():.3f}")
print(nmr.calibrate_coates(nmr.depth[both], k_vendor[both]))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(k_vendor[both], k_library[both], ".", ms=2, alpha=0.3, color="navy")
ax.plot((1e-4, 1e2), (1e-4, 1e2), color="red", lw=1, label="1:1")
ax.set(xlim=(1e-4, 1e2), ylim=(1e-4, 1e2), xlabel="Vendor KTIM (mD)",
       ylabel="Petrophys Timur-Coates (mD)",
       title="Daydream 1: Timur-Coates, library vs vendor")
ax.legend()
plt.show()
```

`permeability_coates()` and `permeability_sdr()` return the vendor curve unless `prefer_stored=False`. This file has no T2 bins, so T2 log-mean and SDR from scratch need a richer delivery. Petrophys does not rebuild distributions from raw echo trains.

## Spectral gamma: 25/5-1

The trace confirms that `pp.interpret` skips it:

```{code-cell} python
from textwrap import fill

well25 = pp.read(pp.sample("25_5-1"))
print(fill(pp.interpret(well25).specific_steps_skipped["spectral_gamma"], 88))
```

{func}`~petrophys.interpretation.geochemistry.interpret_spectral_gamma` finds thorium and uranium, converts units, and adds uranium's share of the gamma ray. The GR mnemonic does not name the spectral tool, so we pass it as the total.

```{code-cell} python
from petrophys.interpretation.geochemistry import interpret_spectral_gamma

gamma = interpret_spectral_gamma(well25, alias_overrides={"SGR": ["GR"]})
print("Inputs used:", gamma.sources)
added = ", ".join(gamma.output_curves)
print(fill(f"Curves added: {added}", 88, subsequent_indent="    "))

depth, gr = well25.depth, well25["GR"].data
th, u = gamma.measurements["THORIUM"], gamma.measurements["URANIUM"]
logged = np.isfinite(gr) & np.isfinite(th) & np.isfinite(u)
# The run ends with held readings (a value repeated exactly, then zeros); stop above.
rows = np.flatnonzero(logged)
held = np.concatenate([rows[1:][np.diff(x[rows]) == 0] for x in (gr, th, u)])
window = logged & (depth < depth[held].min())
u_rich = window & (gamma.derived["SGR_URANIUM_EXCESS_FLAG"] == 1)
rated = window & (u >= 0.5)                           # rate Th/U only where U >= 0.5 ppm
th_u = np.full(depth.shape, np.nan)
th_u[rated] = th[rated] / u[rated]
print(f"Window {depth[window].min():.0f}-{depth[window].max():.0f} m: {u_rich.sum()} of "
      f"{window.sum()} samples get at least half their gamma ray from uranium")

fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
axes[0].plot(np.where(window, gr, np.nan), depth, color="0.25", lw=0.7)
for curve, color, label in [(th, "#eb6834", "thorium"), (u, "#2a78d6", "uranium")]:
    axes[1].plot(np.where(window, curve, np.nan), depth, color=color, lw=0.7, label=label)
for lo, hi, color, label in [(0.1, 2, "#2a78d6", "< 2 marine, reducing"),
                             (2, 7, "0.6", "2-7 marine"),
                             (7, 1000, "#eb6834", "> 7 continental")]:
    axes[2].axvspan(lo, hi, color=color, alpha=0.12, lw=0, label=label)
axes[2].scatter(th_u, depth, s=3, color="0.35", lw=0)
for ax in axes:
    ax.fill_betweenx(depth, 0, 1, where=u_rich, transform=ax.get_yaxis_transform(),
                     color="#2a78d6", alpha=0.12, lw=0)
    ax.grid(alpha=0.3)
axes[0].set(xlabel="GR (API)", ylabel="Depth (m)",
            ylim=(depth[window].max(), depth[window].min()))
axes[1].set(xlabel="Th, U (ppm)")
axes[2].set(xscale="log", xlim=(0.2, 100), xlabel="Th/U")
axes[1].legend(loc="lower right", fontsize=8)
axes[2].legend(loc="lower right", fontsize=8)
fig.suptitle("25/5-1 spectral gamma (blue bands: uranium ≥ 50% of GR)")
plt.tight_layout()
plt.show()
```

Where uranium dominates, GR over-reads clay and thorium is the safer clay indicator. {func}`~petrophys.interpretation.geochemistry.interpret_elemental_spectroscopy` handles elemental yields, also outside `pp.interpret`.

## Core data and calibration: 30/6-5

{func}`pp.attach_core_measurements <petrophys.io.attach_core_measurements>` checks a lab table's units and depths, then puts each plug on the nearest log row. The 30/6-5 plugs, which ship inside its LAS, stand in for a lab table here. That LAS labels its percent core SWC and SOC as fractions; Petrophys converts them with a warning, hidden on this page.

```{code-cell} python
from petrophys.interpretation.core_analysis import (
    apply_poro_perm_transform, calibrate_porosity, core_data_from_well,
    fit_poro_perm_transform,
)

lab = pp.read(pp.sample("30_6-5")).to_dataframe().dropna(subset=["PHIC"])
core_table = pd.DataFrame({
    "well": "30/6-5", "depth": lab["DEPTH"].round(2), "depth_unit": "m",
    "core_porosity": lab["PHIC"], "porosity_unit": "fraction",
    "core_permeability": lab["KH_CORE"], "permeability_unit": "mD",
    "grain_density": lab["GDEN"],
})
logs = pp.read(pp.sample("30_6-5"))
for name in ["PHIC", "KH_CORE", "KV_CORE", "GDEN", "SWC", "SOC"]:
    logs.remove_curve(name)                   # a log-only well
report = pp.attach_core_measurements(logs, core_table)
print(f"{report['input_rows']} plugs attached as {report['curves']}, "
      f"each within {report['alignment_tolerance']:.2f} {report['depth_unit']} of a row")

core = core_data_from_well(logs)
print(core)
```

Calibrate an independent density porosity against the plugs, and fit a poro-perm transform.

```{code-cell} python
from petrophys.visualization import plot_calibration, plot_core_on_logs, plot_poro_perm

rhob = logs["RHOB"].data
logs.add_curve("PHID", np.clip((2.65 - rhob) / (2.65 - 1.0), 0, 0.5), "v/v",
               "Density porosity, quartz matrix")
calibration = calibrate_porosity(logs, core, log_curve="PHID")
transform = fit_poro_perm_transform(core)
print(f"PHID - core: bias {calibration.bias:+.3f} v/v, RMSE {calibration.rmse:.3f}, "
      f"R² {calibration.r_squared:.2f} over {calibration.n_points} plugs")
print(f"log10 k = {transform.A:.1f} φ {transform.B:+.2f}  (R² {transform.r_squared:.2f})")

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
plot_calibration(calibration, ax=left, title="Density porosity against core")
plot_poro_perm(core, transform=transform, ax=right, title="Core poro-perm and transform")
plt.tight_layout()
plt.show()
```

Apply the transform along the log and compare it with the plugs.

```{code-cell} python
apply_poro_perm_transform(logs, transform, porosity_curve="PHID",
                          output_curve="PERM_CORE")
fig, axes = plot_core_on_logs(logs, core, curves=["PHID", "PERM_CORE"], figsize=(8, 7),
                              depth_range=(core.top - 10, core.bottom + 10))
plt.show()
```

Core saturation stays calibration evidence; it does not replace the log Sw curve. Check depth ties first: on 25/5-1 the plugs need a depth shift before they match the logs ({doc}`tutorial 12 <tutorials/part_b/12_core_calibration_and_digital_core>`).

## Saturation-height: HW-3

A Leverett J-function fitted to porosity, permeability and log Sw predicts Sw from height above the free-water level (FWL). HW-3 covers only the Kharaib-2 carbonate. Its FWL is assumed 20 ft below the deepest valid sample, not picked from pressures.

```{code-cell} python
from petrophys.interpretation.saturation.height import (
    FLUID_SYSTEMS, fit_j_function, leverett_j, pc_from_height,
    saturation_height_from_logs,
)
from petrophys.visualization import create_saturation_height_plot

hw3 = pp.read(pp.sample("HW-3"))
table = hw3.to_dataframe()
phi, perm, sw = table["PHIE_CC"].values, table["PERM_CC"].values, table["SW_INPUT"].values
valid = (phi > 0) & (perm > 0) & (sw > 0) & (sw <= 1)
fwl = table["DEPTH"][valid].max() + 20.0                    # ft, assumed
ift, theta = FLUID_SYSTEMS["oil_brine"]
height = np.clip(fwl - table["DEPTH"].values, 0, None)
pc = pc_from_height(height, sg_water=1.0, sg_hc=0.8)
j = leverett_j(pc, perm, phi, ift=ift, theta=theta)
fit = valid & (j > 0)
model = fit_j_function(j[fit], sw[fit], form="power", ift=ift, theta=theta,
                       sg_water=1.0, sg_hc=0.8)
print(f"Sw = {model.a:.3f} J^{model.b:.3f}, R² {model.r2:.2f}, {fit.sum()} samples")

predicted = saturation_height_from_logs(hw3, model, fwl_depth=fwl,
                                        perm_curve=("PERM_CC",), phi_curve=("PHIE_CC",),
                                        depth_increases_down=True)
error = predicted[fit] - sw[fit]
print(f"Model against log Sw: mean absolute error {np.abs(error).mean():.3f}, "
      f"bias {error.mean():+.3f}")

create_saturation_height_plot(model, j=j[fit], sw=sw[fit], max_height_ft=250,
                              rock_classes=[(1, 0.08), (10, 0.15), (100, 0.25)],
                              figsize=(11, 4.5), title="HW-3 Kharaib-2 saturation-height")
plt.show()
```

The low R² is typical of a heterogeneous carbonate with one assumed FWL. Better rock still drains to lower Sw at the same height.

## Borehole images

Petrophys orients, equalizes, measures coverage of and draws image logs; it does not pick dips or fractures. This cell needs `petrophys[dlis]` and a Git LFS image file, so it does not run here.

```{code-cell} python
:tags: [skip-execution]

from petrophys.interpretation.borehole_imaging import (
    azimuthal_coverage, orient_to_azimuth, plot_fmi, read_fmi_dlis,
)
from petrophys.visualization import create_fmi_panel

image = read_fmi_dlis("daydream1_fmi_processed_dynamic_image.dlis")
window = image.filter_by_depth(2900.0, 2910.0)       # cut a window before orienting
oriented = orient_to_azimuth(window, n_azimuth=180, reference="north")
print(f"Imaged circumference: {np.nanmean(azimuthal_coverage(oriented)):.0%}")
plot_fmi(oriented, equalize="static")
create_fmi_panel(image, depth_range=(4052, 4058), orient=True, equalize="dynamic")
```

{doc}`Tutorial 16 <tutorials/part_b/16_borehole_image_fmi>` runs it on the real Daydream 1 FMI.

## Specialist inputs and remaining tutorial work

The Epslog collection now supplies core-scanner inputs for 15/12-20 S and a
same-well RealPore core extract. Discover them with `petrophys.list_samples("epslog")`;
the collection README records curve mappings, units and depth-matching limits.
These inputs and the Phillips reference tables do not by themselves complete
the notebook exercises.

| Data or workflow | Existing API and remaining requirement | Tutorial |
| --- | --- | --- |
| XRF concentrations and mineral proxies | `XRFData`, `create_xrf_panel`; Epslog inputs available, same-well wireline comparison pending | 12 |
| Grain size and Kozeny-Carman | `GrainSizeData`; Epslog P50 and routine-core measurements available, depth/unit verification pending; P13/P87 absent | 12 |
| Scanner strength and dynamic mechanics | `CoreMechanical`; Epslog Vp/Vs available; static moduli, plug-test validation and same-well wireline logs still missing | 12 |
| NMR T2 distributions and SDR | `NMRData` with bins; Phillips teaching example available, traceable field distribution and matching core calibration still missing | 14 |
| Laboratory capillary pressure | `convert_pc_lab_to_reservoir`, `fit_brooks_corey`, `fit_j_function`; COSTA sample curves available, complete fluid/unit and well/depth metadata pending | 15 |

`read_formation_pressure` accepts CSV/LAS files, DataFrames and LAS objects.
Call `.fit_fwl()` to intersect oil/water or gas/water pressure gradients, and
`create_formation_pressure_plot(data, fit)` to plot the stations and result.
The array-level `fit_pressure_fwl` API remains available. Units, a positive-down
TVD datum, fluid labels and measurement QC must be established; column mappings
and numeric fluid-code mappings can be supplied explicitly. Images are not inputs.

Tutorial 15 calls these library APIs for 35/11-13 (tabulated MDT pressures) and
35/11-10 Fensfjord (digitized station markers). CSV and LAS give identical fits:
3113.7 m versus reported 3112.6 m, and 2287.3 m versus reported 2290 m TVD MSL,
respectively. These are interpretation reproduction checks, not blind validation;
the second case also has plot-digitization limitations. Report comparisons are
outside the module. See the `formation_pressure` dataset README for provenance,
quality selection and a custom-input recipe.

Pressure alone does not determine a saturation-height function. Capillary-pressure
or rock-property calibration and fluid properties are still required.
`saturation_height_from_logs` accepts `fwl_depth` as an input on a matching datum
and units; neither of these contacts applies to the separate HW-3 example.

Sonic saturation (`petrophys.interpretation.saturation.sonic`) estimates Sw without resistivity, by Gassmann substitution, and says when the fluid effect is undetectable.

## Go deeper

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorial 12: core calibration
:link: tutorials/part_b/12_core_calibration_and_digital_core
:link-type: doc
Core ties, poro-perm, digital facies and spectral gamma.
:::

:::{grid-item-card} Tutorial 14: NMR
:link: tutorials/part_b/14_nmr_petrophysics
:link-type: doc
The Daydream CMR in depth, with QC of coal and washouts.
:::

:::{grid-item-card} Tutorial 15: saturation-height
:link: tutorials/part_b/15_capillary_pressure_saturation_height
:link-type: doc
Capillary pressure, J-functions and Sw from height.
:::

:::{grid-item-card} Tutorial 16: borehole images
:link: tutorials/part_b/16_borehole_image_fmi
:link-type: doc
Orienting and equalizing the Daydream FMI.
:::
::::
