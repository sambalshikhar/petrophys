---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# What Petrophys can do

You have interpreted a first well and read your own files. The guide continues from there, from interpretation and review to LLM agents; this page maps each part of the library to the page that covers it, then shows what the results look like.

```{code-cell} python
:tags: [remove-cell]

%matplotlib inline
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from myst_nb import glue

from petrophys.visualization import use_plot_style

# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The input templates page explains this.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
use_plot_style()
THUMB = (4, 3)  # inches: every gallery figure is a small thumbnail


def thumb(key, fig):
    """Size a figure as a gallery thumbnail and store it for its card."""
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
    fig.set_size_inches(*THUMB)
    fig.tight_layout()
    glue(key, fig, display=False)
    plt.close(fig)
```

## At a glance

| Area | What you get | Start with | Covered in |
| --- | --- | --- | --- |
| Prepare the data | LAS, DLIS and CSV reading; standard names and units; range QC; bad-hole flags; normalization; ML gap filling | `pp.read`, `pp.impute` | {doc}`own_data`, {doc}`data_model`, {ref}`review-inputs`, {ref}`analysis-gap-filling` |
| Interpret the rock | Temperature, clay volume, porosity, lithology and minerals, Rw, eight Sw models, permeability, rock physics | `pp.interpret` | {doc}`workflows` |
| Review the answer | Warnings, the step-by-step trace, the cutoffs used, net pay per formation | `result.print_trace()`, `result.formation_summary` | {ref}`review-interpretation` |
| Your own recipe | Built-in and custom steps, checked before they run, saved as JSON and run on many wells | `Workflow`, `@workflow_step` | {doc}`custom_workflows` |
| Plots and delivery | Log viewer templates, crossplots, multi-well charts; LAS, JSON, CSV and Excel exports | `pp.plot` | {doc}`visualization_reporting` |
| Core and specialist data | Core calibration, NMR, spectral gamma, saturation-height, borehole images, formation pressures | `core_data_from_well`, `NMRData` | {doc}`specialist_data` |
| Further analyses | Rock-quality and matrix indicators, shale distribution, heterogeneity, zonation, Monte Carlo, invasion, pore pressure | `zone_well`, `monte_carlo` | {doc}`analysis_tools` |
| Many wells | Batch and zoned runs over a folder of wells | `pp.interpret(folder)`, `BatchProcessor` | {ref}`workflows-batch`, {doc}`custom_workflows` |
| LLMs | Bounded JSON context files of interpreted wells; an MCP server that lets a model run Petrophys | `pp.export_context`, `petrophys-mcp` | {doc}`context`, {doc}`mcp` |

The {doc}`complete tour <tutorials/complete_tour>` runs every part of the library on real wells in one notebook.

## What the results look like

Every figure below was computed from the bundled sample wells when these docs were built; the code is at the end of the page.
Depth logs show one representative interval, so the curves can be read.

### Prepare the data

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reading and QC
```{glue:figure} cap_qc
```
Bad hole from caliper against bit size and the density correction: gauge hole above 3060 m on 30/6-5, washed out below.

`pp.read(path)` · `detect_washout(well)`
+++
{doc}`Tutorial 02: QC and standardization <tutorials/part_a/02_qaqc_and_standardization>`
:::

:::{grid-item-card} ML gap filling
```{glue:figure} cap_ml
```
The density of COSTA well HW-24, removed and rebuilt from the other logs by a model trained on the 16 other wells.
The window shown has the well's average error.

`pp.impute(well, "RHOB", train_wells=[...])` adds `RHOB_IMPUTED`.
+++
{doc}`Tutorial 13: ML curve imputation <tutorials/part_b/13_ml_curve_imputation>`
:::
::::

### Interpret the rock

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Clay, porosity and lithology
```{glue:figure} cap_nd
```
Neutron-density crossplot of the 30/6-5 Brent Group, with the matrix lines.

`pp.interpret(well)` · `create_nd_crossplot(well)`
+++
{doc}`Tutorial 05: lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Mineral volumes
```{glue:figure} cap_minerals
```
The mineral solver splits the 15/9-14 chalk into porosity and mineral volumes, mostly calcite.

`MultiMineralSolverV1().solve(...)` (run by `pp.interpret`)
+++
{doc}`Tutorial 05: lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Rw and the Pickett plot
```{glue:figure} cap_pickett
```
Check Rw and the Archie exponents against the 30/6-5 Statfjord water leg.

`create_pickett_plot(well, rw=0.07)`
+++
{doc}`Tutorial 06: formation water <tutorials/part_a/06_formation_water_and_environment>`
:::

:::{grid-item-card} Water-saturation models
```{glue:figure} cap_sw
```
Brent Group net pay under four saturation models, one setting apart.

`pp.interpret(well, sw_model="simandoux")`
+++
{doc}`Tutorial 07: water saturation <tutorials/part_a/07_water_saturation>`
:::
::::

### Evaluate the reservoir

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Permeability from core
```{glue:figure} cap_poroperm
```
A poro-perm transform fitted to the 30/6-5 core plugs, then applied along the log.

`fit_poro_perm_transform(core)`
+++
{doc}`Tutorial 08: permeability <tutorials/part_a/08_permeability>`
:::

:::{grid-item-card} Rock quality and flow units
```{glue:figure} cap_flow
```
The 30/6-5 core plugs grouped into hydraulic flow units by FZI.

`calculate_flow_units(perm, phie)`
+++
{doc}`Tutorial 09: reservoir quality <tutorials/part_a/09_reservoir_quality_and_net_pay>`
:::

:::{grid-item-card} Many wells
```{glue:figure} cap_costa
```
Seventeen COSTA carbonate wells interpreted in one call, ranked by Kharaib-2 net pay.

`pp.interpret(wells)` · `results.summary()`
+++
{doc}`Tutorial 18: multi-well study <tutorials/part_b/18_multiwell_field_study>`
:::
::::

### Core and specialist data

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Core calibration
```{glue:figure} cap_calibration
```
The bias and scatter of 30/6-5 density porosity against core.

`calibrate_porosity(well, core, log_curve="PHID")`
+++
{doc}`Tutorial 12: core calibration <tutorials/part_b/12_core_calibration_and_digital_core>`
:::

:::{grid-item-card} NMR
```{glue:figure} cap_nmr
```
Timur-Coates permeability from the Daydream CMR, checked against the vendor's.

`NMRData.from_las(path)` · `nmr.permeability_coates()`
+++
{doc}`Tutorial 14: NMR <tutorials/part_b/14_nmr_petrophysics>`
:::

:::{grid-item-card} Spectral gamma
```{glue:figure} cap_sgr
```
Thorium and uranium on 25/5-1, with the uranium-dominated samples flagged.

`interpret_spectral_gamma(well)`
+++
{doc}`Tutorial 12, part 3 <tutorials/part_b/12_core_calibration_and_digital_core>`
:::

:::{grid-item-card} Saturation-height
```{glue:figure} cap_height
```
A J-function fitted on HW-3 predicts Sw from height above the free-water level and rock quality.

`fit_j_function(j, sw)`
+++
{doc}`Tutorial 15: saturation-height <tutorials/part_b/15_capillary_pressure_saturation_height>`
:::
::::

Borehole images (FMI) are oriented, equalized and displayed from DLIS or LAS arrays; they need `petrophys[dlis]` and a 112 MB file, so they are shown in {doc}`Tutorial 16 <tutorials/part_b/16_borehole_image_fmi>` and not built here.

## What Petrophys does not do

- No automatic depth matching between runs, or tool-depth correction.
- DLIS: processed channels, not every raw acquisition product.
- Borehole images: preparation and display, not dip or fracture picking.
- NMR: delivered distributions or curves, not raw echo trains.
- Empirical models still need local calibration and expert review.

## How the gallery is built

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: reading, QC and gap filling"
  code_prompt_hide: "Hide the code: reading, QC and gap filling"
---
import matplotlib.pyplot as plt
import numpy as np
import petrophys as pp
from petrophys.preprocessing import detect_washout

raw = pp.read(pp.sample("30_6-5"))
washout = detect_washout(raw)  # caliper against bit size, plus density correction
depth = raw.depth
window = (depth >= 3000) & (depth <= 3110)  # gauge hole into a washout
fig, (ax_gr, ax_cal) = plt.subplots(1, 2, sharey=True)
ax_gr.plot(raw["GR"].data[window], depth[window], color="#1b7837", lw=0.6)
ax_cal.plot(raw["CALI"].data[window], depth[window], color="black", lw=0.7, label="CALI")
ax_cal.plot(raw["BS"].data[window], depth[window], color="tab:blue", ls="--", label="bit size")
for ax in (ax_gr, ax_cal):
    ax.fill_betweenx(depth[window], 0, 1, where=washout.bad_hole_flag[window],
                     transform=ax.get_yaxis_transform(), color="tab:red", alpha=0.25, lw=0)
ax_gr.set(xlabel="GR (API)", ylabel="Depth (m)")
ax_cal.set(xlabel="Caliper (in)")
ax_cal.legend(loc="lower right", fontsize=7)
ax_gr.invert_yaxis()
fig.suptitle("30/6-5: bad hole shaded red")
thumb("cap_qc", fig)

costa = [name for name in pp.list_samples() if name.startswith("HW-")]
hw24 = pp.read(pp.sample("HW-24"))
measured = hw24["RHOB"].data.copy()
hw24.remove_curve("RHOB")
pp.impute(hw24, "RHOB", train_wells=[pp.sample(n) for n in costa if n != "HW-24"])
rebuilt = hw24["RHOB_IMPUTED"].data
error = np.abs(rebuilt - measured)
shown = (hw24.depth >= 8510) & (hw24.depth <= 8580)  # ft
print(f"HW-24 density error: {np.nanmean(error):.3f} g/cm3 over the well, "
      f"{np.nanmean(error[shown]):.3f} g/cm3 in the window shown")
fig, ax = plt.subplots()
ax.plot(measured[shown], hw24.depth[shown], color="#9aa5b1", lw=2.2, label="Measured")
ax.plot(rebuilt[shown], hw24.depth[shown], color="#176b87", lw=1, label="Rebuilt")
ax.invert_yaxis()
ax.set(xlabel="RHOB (g/cm3)", ylabel=f"Depth ({hw24.depth_unit})", title="HW-24 density from 16 wells")
ax.legend(loc="lower left", fontsize=8)
thumb("cap_ml", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: crossplot, minerals, Pickett plot and Sw models"
  code_prompt_hide: "Hide the code: crossplot, minerals, Pickett plot and Sw models"
---
from petrophys.visualization import create_bar_comparison, create_nd_crossplot, create_pickett_plot

BRENT = (2841, 2958)  # Brent Group reservoir on 30/6-5, m
well = pp.read(
    pp.sample("30_6-5"),
    formations=pp.sample("formations_30_6-5"),
    lithology=pp.sample("lithology_30_6-5"),
)
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)  # Rw 0.07 ohm.m at 230 degF
interpreted = result.well

fig = create_nd_crossplot(interpreted, color_curve="GR", depth_range=BRENT, figsize=THUMB)
thumb("cap_nd", fig)

chalk = pp.interpret(pp.read(pp.sample("15_9-14")))
seg = chalk.well.depth_slice(2840, 2900)
solved = np.isfinite(seg["VQUAR"].data)
fig, (ax_gr, ax_vol) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 2]})
ax_gr.plot(seg["GR"].data, seg.depth, color="#1b7837", lw=0.7)
left, handles = np.zeros(seg.n_samples), []
for name, label, color in [("VCL", "clay", "#8c6d31"), ("VQUAR", "quartz", "#f2d16b"),
                           ("VCALC", "calcite", "#6baed6"), ("VDOLO", "dolomite", "#9e9ac8"),
                           ("VANHY", "anhydrite", "#f768a1"), ("PHIE", "porosity", "white")]:
    volume = np.nan_to_num(seg[name].data)
    ax_vol.fill_betweenx(seg.depth, left, left + volume, where=solved, step="mid", color=color, lw=0)
    left = left + volume
    if volume.max() > 0.01:  # the legend names what is present
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="0.4", lw=0.5, label=label))
if not solved.all():
    ax_vol.set_facecolor("0.8")  # grey: samples the solver could not constrain
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.8", label="unsolved"))
ax_gr.set(xlabel="GR (API)", ylabel="Depth (m)", ylim=(seg.depth.max(), seg.depth.min()))
ax_vol.set(xlim=(0, 1), xlabel="Volume (v/v)")
ax_vol.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              fontsize=7, handlelength=1, columnspacing=0.8)
fig.suptitle("15/9-14 chalk: mineral volumes")
thumb("cap_minerals", fig)

pickett = create_pickett_plot(
    interpreted, rw=0.07, rt_curve="RDEP", color_curve="VCL",
    depth_range=(3223, 3556), a=0.8, m=2.0, n=2.0, figsize=THUMB,  # Statfjord water leg
)
thumb("cap_pickett", pickett.fig)

net_pay = {}
# Dual Water needs a measured bound-water resistivity (rwb), and Waxman-Smits
# needs CEC; Petrophys does not guess them, so they are left out here.
for model in ["archie", "simandoux", "indonesia", "auto"]:
    run = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model=model)
    net_pay[model] = run.summary(top=BRENT[0], bottom=BRENT[1])["net_thickness"]["net_pay"]
fig, ax = create_bar_comparison(list(net_pay), list(net_pay.values()), value_labels=True,
                                ylabel="Brent Group net pay (m)", title="Net pay by Sw model, 30/6-5",
                                figsize=THUMB)
ax.set_ylim(0, 1.2 * max(net_pay.values()))
thumb("cap_sw", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: core, flow units and many wells"
  code_prompt_hide: "Hide the code: core, flow units and many wells"
---
from petrophys.evaluation import calculate_flow_units
from petrophys.interpretation.core_analysis import (
    calibrate_porosity, core_data_from_well, fit_poro_perm_transform,
)
from petrophys.visualization import create_flow_unit_plot, plot_calibration, plot_poro_perm

core = core_data_from_well(interpreted)  # PHIC, KH_CORE, GDEN, SWC plugs in the LAS
transform = fit_poro_perm_transform(core)
fig, ax = plot_poro_perm(core, transform=transform, title="30/6-5 core poro-perm", figsize=THUMB)
ax.legend(loc="lower right")
thumb("cap_poroperm", fig)

plugs = np.isfinite(core.porosity) & np.isfinite(core.permeability)
units = calculate_flow_units(core.permeability[plugs], core.porosity[plugs])
thumb("cap_flow", create_flow_unit_plot(result=units, figsize=THUMB, title="30/6-5 flow units"))

calibration = calibrate_porosity(interpreted, core, log_curve="PHID")
fig, ax = plot_calibration(calibration, title="Density porosity vs core", figsize=THUMB)
thumb("cap_calibration", fig)

# Seventeen COSTA wells, each covering the Kharaib-2 interval only.
costa_wells = pp.read([pp.sample(name) for name in pp.list_samples("costa")],
                      formations=pp.sample("formations_costa"))
field = pp.interpret(costa_wells, lithology_domain="carbonate",
                     vcl_cutoff=0.30, phie_cutoff=0.10, sw_cutoff=0.60)
pay = field.summary().set_index("Well")["Net pay"].sort_values(ascending=False)
fig, ax = create_bar_comparison(list(pay.index), list(pay.values), figsize=THUMB,
                                ylabel="Net pay (ft)", title="Kharaib-2 net pay, COSTA field")
thumb("cap_costa", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: NMR, spectral gamma and saturation-height"
  code_prompt_hide: "Hide the code: NMR, spectral gamma and saturation-height"
---
from petrophys.interpretation.geochemistry import interpret_spectral_gamma
from petrophys.interpretation.nmr import NMRData
from petrophys.interpretation.saturation.height import (
    FLUID_SYSTEMS, fit_j_function, leverett_j, pc_from_height,
)

# NMR: library Timur-Coates against the vendor KTIM curve.
nmr = NMRData.from_las(pp.sample("daydream1_cmr"), mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
k_lib = nmr.permeability_coates(prefer_stored=False)
k_vendor = nmr.stored_permeability["KTIM"]
both = (k_lib > 0) & (k_vendor > 0)
fig, ax = plt.subplots()
ax.loglog(k_vendor[both], k_lib[both], ".", ms=2, alpha=0.3, color="navy")
ax.plot((1e-4, 1e2), (1e-4, 1e2), color="red", lw=1, label="1:1")
ax.set(xlim=(1e-4, 1e2), ylim=(1e-4, 1e2), xlabel="Vendor KTIM (mD)",
       ylabel="Petrophys Timur-Coates (mD)", title="Daydream CMR permeability")
ax.legend()
thumb("cap_nmr", fig)

# Spectral gamma: pp.interpret does not run it, so call it directly.
gamma = interpret_spectral_gamma(pp.read(pp.sample("25_5-1")), alias_overrides={"SGR": ["GR"]})
th, u = gamma.measurements["THORIUM"], gamma.measurements["URANIUM"]
u_rich = gamma.derived["SGR_URANIUM_EXCESS_FLAG"] == 1
shown = (th > 0.1) & (u > 0.1)
fig, ax = plt.subplots()
ax.scatter(u[shown & ~u_rich], th[shown & ~u_rich], s=3, color="0.6", lw=0, label="other samples")
ax.scatter(u[shown & u_rich], th[shown & u_rich], s=4, color="#2a78d6", lw=0,
           label="uranium ≥ 50% of GR")
line = np.geomspace(0.1, 50, 20)
ax.plot(line, 2 * line, "k--", lw=0.8, label="Th/U = 2")
ax.plot(line, 7 * line, "k:", lw=0.8, label="Th/U = 7")
ax.set(xscale="log", yscale="log", xlim=(0.1, 50), ylim=(0.5, 30),
       xlabel="Uranium (ppm)", ylabel="Thorium (ppm)", title="25/5-1 spectral gamma")
ax.legend(loc="lower right")
thumb("cap_sgr", fig)

# Saturation-height: fit a J-function on HW-3 (Kharaib-2). The FWL is assumed
# 1 ft below the deepest valid sample, the base of the transition zone.
hw3 = pp.read(pp.sample("HW-3")).to_dataframe()
phi, perm, sw = hw3["PHIE_CC"].values, hw3["PERM_CC"].values, hw3["SW_INPUT"].values
fwl = hw3["DEPTH"][np.isfinite(phi * perm * sw)].max() + 1.0  # ft, assumed FWL
height = np.clip(fwl - hw3["DEPTH"].values, 0, None)
ift, theta = FLUID_SYSTEMS["oil_brine"]
j = leverett_j(pc_from_height(height, sg_water=1.0, sg_hc=0.8), perm, phi, ift=ift, theta=theta)
fit = (j > 0) & (sw > 0) & (sw <= 1)
model = fit_j_function(j[fit], sw[fit], ift=ift, theta=theta, sg_water=1.0, sg_hc=0.8)
fig, ax = plt.subplots()
points = ax.scatter(sw[fit], height[fit], c=np.log10(perm[fit]), s=6, lw=0, cmap="viridis")
h = np.linspace(0.5, height[fit].max(), 200)
for k, p, color in [(1, 0.10, "#440154"), (10, 0.18, "#21918c"), (100, 0.25, "#b5de2b")]:
    ax.plot(model.sw_from_height(h, np.full_like(h, k), np.full_like(h, p)), h,
            color=color, lw=1.6, label=f"model, {k} mD")
fig.colorbar(points, ax=ax, label="log10 perm (mD)")
ax.set(xlim=(0, 1), xlabel="Sw (v/v)", ylabel="Height above FWL (ft)", title="HW-3 saturation-height")
ax.legend(loc="upper right")
thumb("cap_height", fig)
```

## Next

{doc}`workflows`: interpret a well with `pp.interpret`, read its trace and review the answer.
