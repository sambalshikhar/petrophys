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
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from myst_nb import glue

from petrophys.visualization import use_plot_style

# The gallery shows figures only; the pages its cards link to show each run's warnings.
warnings.simplefilter("ignore")
use_plot_style()
THUMB, LOG_THUMB = (4, 3), (4, 4)  # inches: gallery thumbnails, and log-plot thumbnails


def thumb(key, fig):
    """Store a figure as the thumbnail of its gallery card."""
    for ax in fig.axes:
        ax.xaxis.set_minor_formatter(NullFormatter())  # no crowded minor labels on log axes
    fig.tight_layout()
    glue(key, fig, display=False)
    plt.close(fig)
```

## At a glance

| Area | What you get | Start with | Covered in |
| --- | --- | --- | --- |
| Prepare the data | LAS, DLIS, CSV and Excel reading; standard names and units; range QC; bad-hole flags; normalization; ML gap filling | `pp.read`, `pp.impute` | {doc}`own_data`, {doc}`data_model`, {ref}`review-inputs`, {ref}`analysis-gap-filling` |
| Interpret the rock | Temperature, clay volume, porosity, lithology and minerals, Rw, eight Sw models, permeability, rock physics | `pp.interpret` | {doc}`workflows` |
| Review the answer | Warnings, the step-by-step trace, the cutoffs used, net pay per formation | `result.print_trace()`, `result.formation_summary()` | {ref}`review-interpretation` |
| Your own recipe | Built-in and custom steps, checked before they run, saved as JSON and run on many wells | `Workflow`, `@workflow_step` | {doc}`custom_workflows` |
| Plots and delivery | Log viewer templates, crossplots, multi-well charts; LAS, JSON, CSV and Excel exports | `pp.plot` | {doc}`visualization_reporting` |
| Core and specialist data | Core calibration, NMR, spectral gamma, saturation-height, borehole images, formation pressures | `core_data`, `NMRData` | {doc}`specialist_data` |
| Further analyses | Rock-quality and matrix indicators, shale distribution, heterogeneity, zonation, Monte Carlo, invasion, pore pressure | `zone_well`, `monte_carlo` | {doc}`analysis_tools` |
| Many wells | Batch and zoned runs over a folder of wells | `pp.interpret(folder)`, `BatchProcessor` | {ref}`workflows-batch`, {doc}`custom_workflows` |
| LLMs | Bounded JSON context files of interpreted wells; an MCP server that lets a model run Petrophys | `pp.export_context`, `petrophys-mcp` | {doc}`context`, {doc}`mcp` |

The {doc}`complete tour <tutorials/complete_tour>` runs every part of the library on real wells in one notebook.

## What the results look like

Every figure below was computed from the bundled sample wells when these docs were built, by the call under it; the code is at the end of the page.
Depth logs show one representative interval, so the curves can be read.

### Prepare the data

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reading and QC
```{glue:figure} cap_qc
```
Bad hole (shaded) from caliper against bit size and the density correction: gauge hole above 3060 m on 30/6-5, washed out below.

`pp.plot(pp.interpret(well), curves=["GR", "CALI", "BS"])`
+++
{doc}`Tutorial 02: QC and standardization <tutorials/part_a/02_qaqc_and_standardization>`
:::

:::{grid-item-card} ML gap filling
```{glue:figure} cap_ml
```
The density of COSTA well HW-24 (red) against its prediction from the other logs (black), by a model trained on this well and the 16 other wells. Each measured depth is predicted by a model that did not learn from that fifth of the well: mean absolute error 0.035 g/cm3, R² 0.89 over 433 samples.

`pp.impute(well, "RHOB", train_wells=[...])` · `pp.plot(well, curves=["RHOB", "RHOB_PREDICTED"])`
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

`pp.crossplot(well, "NPHI", "RHOB")`
+++
{doc}`Tutorial 05: lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Mineral volumes
```{glue:figure} cap_minerals
```
The mineral solver splits the 15/9-14 chalk into porosity and mineral volumes, mostly calcite.

`pp.plot(pp.interpret(well), template="minerals")`
+++
{doc}`Tutorial 05: lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Rw and the Pickett plot
```{glue:figure} cap_pickett
```
Check Rw and the Archie exponents against the 30/6-5 Statfjord water leg.

`pp.crossplot(result, "PHIE", "RDEP")`
+++
{doc}`Tutorial 06: formation water <tutorials/part_a/06_formation_water_and_environment>` · {doc}`Tutorial 07: the Pickett plot <tutorials/part_a/07_water_saturation>`
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
A poro-perm transform fitted to the 30/6-5 core plugs: permeability on a log scale against porosity, coloured by depth.

`pp.crossplot(core, "PHIC", "KH_CORE")`
+++
{doc}`Tutorial 08: permeability <tutorials/part_a/08_permeability>`
:::

:::{grid-item-card} Rock quality and flow units
```{glue:figure} cap_flow
```
The 30/6-5 core plugs grouped into hydraulic flow units by FZI.

`pp.crossplot(core_data(well), "PHI_Z", "RQI")`
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

`calibration_plot(calibrate_porosity(well, core, log_curve="PHID"))`
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

`interpret_spectral_gamma(well)` · `pp.crossplot(well, "URANIUM", "THORIUM")`
+++
{doc}`Tutorial 12, part 3 <tutorials/part_b/12_core_calibration_and_digital_core>`
:::

:::{grid-item-card} Saturation-height
```{glue:figure} cap_height
```
A J-function fitted to the log Sw of HW-3 above an assumed free-water level (7991 ft, 1 ft below the deepest sample that has Sw) predicts Sw from height and rock quality; the fit is loose (R² 0.28).

`fit_j_function_from_well(well, fwl_depth=7991).plot()`
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

Each card's figure comes from the call under it. The NMR figure is drawn with
matplotlib from the library's results.

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: reading, QC, interpretation and Sw models"
  code_prompt_hide: "Hide the code: reading, QC, interpretation and Sw models"
---
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"),
               lithology=pp.sample("lithology_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp="230 degF")   # Rw 0.07 ohm.m at 230 degF

thumb("cap_qc", pp.plot(result, top=3000, bottom=3110, curves=["GR", "CALI", "BS"],
                        figsize=LOG_THUMB).fig)
thumb("cap_nd", pp.crossplot(well, "NPHI", "RHOB", top=2841, bottom=2958, figsize=THUMB).fig)
chalk = pp.interpret(pp.read(pp.sample("15_9-14")))
thumb("cap_minerals", pp.plot(chalk, top=2840, bottom=2900, template="minerals",
                              figsize=LOG_THUMB).fig)
thumb("cap_pickett", pp.crossplot(result, "PHIE", "RDEP", top=3223, bottom=3556, a=0.8, m=2.0, n=2.0,
                                  figsize=THUMB).fig)

# Dual Water needs a bound-water resistivity (rwb) and Waxman-Smits a CEC;
# 30/6-5 gives neither, so they are left out.
models = ["archie", "simandoux", "indonesia", "auto"]
pay = [pp.interpret(well, rw=0.07, rw_meas_temp="230 degF", sw_model=model)
       .summary(top=2841, bottom=2958)["net_thickness"]["net_pay"] for model in models]
fig, ax = pp.bar_chart(models, pay, value_labels=True, figsize=THUMB,
                       ylabel="Brent Group net pay [m]", title="Net pay by Sw model, 30/6-5")
ax.set_ylim(0, 1.2 * max(pay))
thumb("cap_sw", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: core, flow units, spectral gamma and many wells"
  code_prompt_hide: "Hide the code: core, flow units, spectral gamma and many wells"
---
from petrophys.interpretation.core_analysis import (
    calibrate_porosity, core_data,
)
from petrophys.interpretation.geochemistry import interpret_spectral_gamma

core = core_data(result)  # the PHIC, KH_CORE, GDEN and SWC plugs in the LAS
thumb("cap_poroperm", pp.crossplot(core, "PHIC", "KH_CORE", title="30/6-5 core poro-perm", figsize=THUMB).fig)
thumb("cap_flow", pp.crossplot(core, "PHI_Z", "RQI", title="30/6-5 flow units", figsize=THUMB).fig)
fig, ax = pp.calibration_plot(calibrate_porosity(result, core, log_curve="PHID"),
                              title="Density porosity vs core", figsize=THUMB)
thumb("cap_calibration", fig)

gamma_well = pp.read(pp.sample("25_5-1"))
interpret_spectral_gamma(gamma_well, alias_overrides={"SGR": ["GR"]})  # adds THORIUM, URANIUM, ...
thumb("cap_sgr", pp.crossplot(gamma_well, "URANIUM", "THORIUM", color="SGR_URANIUM_EXCESS_FLAG",
                              x_log=True, y_log=True, title="25/5-1 spectral gamma", figsize=THUMB).fig)

# Seventeen COSTA wells, each covering the Kharaib-2 interval only.
field = pp.interpret(pp.read([pp.sample(name) for name in pp.list_samples("costa")],
                             formations=pp.sample("formations_costa")),
                     lithology_domain="carbonate", vcl_cutoff=0.30, phie_cutoff=0.10, sw_cutoff=0.60)
pay = field.summary().set_index("Well")["Net pay"].sort_values(ascending=False)
fig, ax = pp.bar_chart(list(pay.index), list(pay.values), figsize=THUMB,
                       ylabel="Net pay [ft]", title="Kharaib-2 net pay, COSTA field")
thumb("cap_costa", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: gap filling, NMR and saturation-height"
  code_prompt_hide: "Hide the code: gap filling, NMR and saturation-height"
---
from petrophys.interpretation.nmr import NMRData
from petrophys.interpretation.saturation.height import fit_j_function_from_well

# Gap filling: HW-24 density predicted at every depth, out of fold, with the 16 other wells.
hw24 = pp.read(pp.sample("HW-24"))
pp.impute(hw24, "RHOB", train_wells=[pp.sample(name) for name in pp.list_samples("costa")
                                     if name != "HW-24"])
thumb("cap_ml", pp.plot(hw24, curves=["RHOB", "RHOB_PREDICTED"], top=8510, bottom=8580,
                        figsize=LOG_THUMB).fig)

# NMR: the library's Timur-Coates permeability against the vendor KTIM curve.
nmr = NMRData.from_las(pp.sample("daydream1_cmr"), mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
fig, ax = plt.subplots(figsize=THUMB)
ax.loglog(nmr.stored_permeability["KTIM"], nmr.permeability_coates(prefer_stored=False),
          ".", ms=2, alpha=0.3, color="navy")
ax.plot((1e-4, 1e2), (1e-4, 1e2), color="red", lw=1, label="1:1")
ax.set(xlim=(1e-4, 1e2), ylim=(1e-4, 1e2), xlabel="Vendor KTIM (mD)",
       ylabel="Petrophys Timur-Coates (mD)", title="Daydream CMR permeability")
ax.legend()
thumb("cap_nmr", fig)

# Saturation-height: a J-function fitted on HW-3 (Kharaib-2), with the FWL
# taken 1 ft below the deepest sample with Sw, porosity and permeability (7990 ft), the base of the transition zone.
model = fit_j_function_from_well(pp.read(pp.sample("HW-3")), fwl_depth=7991, sw_curve="SW_INPUT",
                                 perm_curve="PERM_CC", phi_curve="PHIE_CC")
fig = model.plot()
fig.set_size_inches(*THUMB)
thumb("cap_height", fig)
```

## Next

{doc}`workflows`: interpret a well with `pp.interpret`, read its trace and review the answer.
