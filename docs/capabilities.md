---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# What Petrophys can do

Petrophys takes well logs, core and specialist measurements to a traced petrophysical interpretation, from reading to net pay.
Every figure and table on this page was computed from the bundled sample wells when these docs were built.

```{code-cell} python
:tags: [remove-cell]

%matplotlib inline
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter
from myst_nb import glue

# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The page explains this once, below.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
pd.set_option("display.max_colwidth", None)

THUMB = (4, 3)  # inches: every gallery figure is a small thumbnail
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7})


def thumb(key, fig):
    """Size a figure as a gallery thumbnail and store it for its card."""
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
    fig.set_size_inches(*THUMB)
    fig.tight_layout()
    glue(key, fig, display=False)
    plt.close(fig)


def table(key, frame):
    """Store a compact table for its card."""
    glue(key, frame.style.hide(axis="index").format(precision=2, na_rep="–"), display=False)
```

## At a glance

| Area | What you get | Start with |
| --- | --- | --- |
| {ref}`Prepare the data <cap-prepare>` | LAS, DLIS and CSV reading; standard names and units; range QC; bad-hole flags; normalization; ML gap filling | `pp.read` |
| {ref}`Interpret the rock <cap-interpret>` | Temperature, clay volume, porosity, lithology and minerals, Rw, seven Sw models, permeability, rock physics | `pp.interpret` |
| {ref}`Evaluate the reservoir <cap-evaluate>` | Core-tied permeability, rock quality and flow units, BVW and moveable hydrocarbons, net pay, formation summaries | `result.formation_summary` |
| {ref}`Core and specialist data <cap-specialist>` | Core calibration, NMR, spectral gamma, saturation-height, borehole images | {doc}`specialist_data` |
| {ref}`Many wells, traces and LLMs <cap-scale>` | Batch and zoned runs, the step-by-step trace, LLM context files, an MCP server | `BatchProcessor`, `pp.export_context` |
| Plots and delivery | Log viewer templates, crossplots, LAS, JSON, CSV and Excel exports | {doc}`visualization_reporting` |

(cap-prepare)=
## Prepare the data

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reading and QC
```{glue:figure} cap_qc
```
Standard names and units, range checks, and bad-hole flags from caliper and bit size.

`pp.read(path)` · `detect_washout(well)`
+++
{doc}`Tutorial 02: QC and standardization <tutorials/part_a/02_qaqc_and_standardization>`
:::

:::{grid-item-card} ML gap filling
```{glue:figure} cap_ml
```
Pretrained models fill only missing DTC, NPHI or RHOB samples; above, their held-out-well scores.

`pp.impute_missing_curve(well, "RHOB")` fills the gaps in place. It needs
`petrophys[ml]`, so it does not run here.
+++
{doc}`Tutorial 13: ML curve imputation <tutorials/part_b/13_ml_curve_imputation>`
:::
::::

(cap-interpret)=
## Interpret the rock

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Clay, porosity and lithology
```{glue:figure} cap_nd
```
Neutron-density crossplot of the Brent reservoir, with matrix lines.

`pp.interpret(well)` · `create_nd_crossplot(well)`
+++
{doc}`Tutorials 03–05 <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Mineral volumes
```{glue:figure} cap_minerals
```
The mineral solver splits the 15/9-14 chalk into calcite, quartz and porosity.

`MultiMineralSolver().solve(...)` (run by `pp.interpret`)
+++
{doc}`Tutorial 05: lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
:::

:::{grid-item-card} Rw and the Pickett plot
```{glue:figure} cap_pickett
```
Check Rw and the Archie exponents against the Statfjord water leg.

`create_pickett_plot(well, rw=0.07)`
+++
{doc}`Tutorial 06: formation water <tutorials/part_a/06_formation_water_and_environment>`
:::

:::{grid-item-card} Water-saturation models
```{glue:figure} cap_sw
```
Brent net pay under five saturation models, one setting apart.

`pp.interpret(well, sw_model="simandoux")`
+++
{doc}`Tutorial 07: water saturation <tutorials/part_a/07_water_saturation>`
:::
::::

(cap-evaluate)=
## Evaluate the reservoir

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Permeability from core
```{glue:figure} cap_poroperm
```
Fit a poro-perm transform to core plugs and apply it along the log.

`fit_poro_perm_transform(core)`
+++
{doc}`Tutorial 08: permeability <tutorials/part_a/08_permeability>`
:::

:::{grid-item-card} Rock quality and flow units
```{glue:figure} cap_flow
```
Group the 30/6-5 core plugs into hydraulic flow units by FZI.

`calculate_flow_units(perm, phie)`
+++
{doc}`Tutorial 09: reservoir quality <tutorials/part_a/09_reservoir_quality_and_net_pay>`
:::

:::{grid-item-card} Net pay by formation
```{glue} cap_formations
```
Net pay, with average porosity and Sw over the pay, per formation (not group) from the Brent down.

`result.formation_summary()`
+++
{doc}`Tutorial 11: formation analysis <tutorials/part_b/11_formation_and_interval_analysis>`
:::
::::

(cap-specialist)=
## Core and specialist data

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Core calibration
```{glue:figure} cap_calibration
```
Measure the bias and scatter of log porosity against core.

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
A J-function fitted on HW-3 predicts Sw from height and rock quality.

`fit_j_function(j, sw)`
+++
{doc}`Tutorial 15: saturation-height <tutorials/part_b/15_capillary_pressure_saturation_height>`
:::

:::{grid-item-card} Borehole images
Orient, equalize and display processed FMI (resistivity image) arrays from DLIS or LAS.

`read_fmi_dlis(path)` · `create_fmi_panel(image)`

They need `petrophys[dlis]` and a Git LFS file, so they do not run here.
+++
{doc}`Tutorial 16: borehole images <tutorials/part_b/16_borehole_image_fmi>`
:::
::::

(cap-scale)=
## Many wells, traces and LLMs

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Multi-well studies
```{glue:figure} cap_costa
```
Interpret all 17 COSTA carbonate wells in one call and rank them.

`BatchProcessor(config=cutoffs)` · `.process_wells(wells)`
+++
{doc}`Tutorial 18: multi-well study <tutorials/part_b/18_multiwell_field_study>`
:::

:::{grid-item-card} The workflow trace
```{glue} cap_trace
```
Each step records its method, inputs and outputs; skipped steps say why.

`result.print_trace()` · `result.to_json(path)`
+++
{doc}`Tutorial 10: the glass-box workflow <tutorials/part_b/10_dynamic_workflow_glass_box>`
:::

:::{grid-item-card} Context export and MCP
```{glue} cap_llm
```
Give an LLM the interpretation as bounded JSON, or Petrophys itself as MCP tools.

`pp.export_context(result, path)` · `petrophys-mcp`
+++
{doc}`Tutorials 19 <tutorials/part_b/19_llm_context_generation>` and {doc}`20 <tutorials/part_b/20_mcp_server>`
:::
::::

## What Petrophys does not do

- No automatic depth matching between runs, or tool-depth correction.
- DLIS: processed channels, not every raw acquisition product.
- Borehole images: preparation and display, not dip or fracture picking.
- NMR: delivered distributions or curves, not raw echo trains.
- Empirical models still need local calibration and expert review.

## How the gallery is built

Each cell below makes one card. A hidden cell sizes the thumbnails. It also hides one warning: 30/6-5 labels its percent core SWC and SOC "frac", and Petrophys converts them.

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the shared setup"
  code_prompt_hide: "Hide the shared setup"
---
import warnings
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petrophys as pp

BRENT = (2841, 2958)  # Brent Group reservoir on 30/6-5, m
well = pp.read(
    pp.sample("30_6-5"),
    formations=pp.sample("formations_30_6-5"),
    lithology=pp.sample("lithology_30_6-5"),
)
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)  # Rw 0.07 ohm.m at 230 degF
interpreted = result.well
print(result.status, "|", interpreted.n_curves, "curves after interpretation")
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: reading and QC"
  code_prompt_hide: "Hide the code: reading and QC"
---
from petrophys.preprocessing import apply_value_range_qc, detect_washout

raw = pp.read(pp.sample("30_6-5"))
fixed = {r["curve"]: r["changed"] for r in apply_value_range_qc(raw) if r.get("changed")}
washout = detect_washout(raw)  # caliper against bit size, plus density correction
print("Values set to null by range QC:", fixed)
print(f"Bad hole: {washout.statistics['bad_hole_percent']:.0f}% of the logged interval")

depth = raw.depth
window = (depth >= 1600) & (depth <= 2500)
fig, (ax_gr, ax_cal) = plt.subplots(1, 2, sharey=True)
ax_gr.plot(raw["GR"].data[window], depth[window], color="#1b7837", lw=0.5)
ax_cal.plot(raw["CALI"].data[window], depth[window], color="black", lw=0.6, label="CALI")
ax_cal.plot(raw["BS"].data[window], depth[window], color="tab:blue", ls="--", label="bit size")
for ax in (ax_gr, ax_cal):
    ax.fill_betweenx(depth[window], 0, 1, where=washout.bad_hole_flag[window],
                     transform=ax.get_yaxis_transform(), color="tab:red", alpha=0.25, lw=0)
ax_gr.set(xlabel="GR (API)", ylabel="Depth (m)")
ax_cal.set(xlabel="Caliper (in)")
ax_cal.legend(loc="lower right")
ax_gr.invert_yaxis()
fig.suptitle("30/6-5: bad hole shaded red")
thumb("cap_qc", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: ML gap filling"
  code_prompt_hide: "Hide the code: ML gap filling"
---
import json

from petrophys.preprocessing.imputation import default_pretrained_model_path

# The validation scores ship as plain JSON beside the model pack.
scores = json.loads(default_pretrained_model_path().with_name("validation.json").read_text())
best = {
    target: max(models.values(), key=lambda m: m["r2"])["r2"]
    for target, models in ((t, v["models"]) for t, v in scores["targets"].items())
}
fig, ax = plt.subplots()
ax.barh(list(best), list(best.values()), color="#176b87")
for y, value in enumerate(best.values()):
    ax.text(value + 0.01, y, f"{value:.2f}", va="center")
ax.set(xlim=(0, 1), xlabel="Best held-out-well R²", title="Bundled imputation models")
thumb("cap_ml", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: crossplot and minerals"
  code_prompt_hide: "Hide the code: crossplot and minerals"
---
from petrophys.visualization import create_nd_crossplot

fig = create_nd_crossplot(interpreted, color_curve="GR", depth_range=BRENT, figsize=THUMB)
thumb("cap_nd", fig)

# 15/9-14 has a chalk section. Its DTE curve is implausible, so Petrophys
# removes it and raises a DataQualityWarning, printed here.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", pp.DataQualityWarning)
    chalk = pp.interpret(pp.read(pp.sample("15_9-14")))
for note in caught:
    print(fill(f"{note.category.__name__}: {note.message}", 88, subsequent_indent="    "))

seg = chalk.well.depth_slice(2880, 2930)
solved = np.isfinite(seg["VQUAR"].data)
fig, (ax_gr, ax_vol) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 2]})
ax_gr.plot(seg["GR"].data, seg.depth, color="#1b7837", lw=0.7)
left = np.zeros(seg.n_samples)
for name, label, color in [("VCL", "clay", "#8c6d31"), ("VQUAR", "quartz", "#f2d16b"),
                           ("VCALC", "calcite", "#6baed6"), ("VDOLO", "dolomite", "#9e9ac8"),
                           ("VANHY", "anhydrite", "#f768a1"), ("PHIE", "porosity", "white")]:
    volume = np.nan_to_num(seg[name].data)
    ax_vol.fill_betweenx(seg.depth, left, left + volume, where=solved, step="mid",
                         color=color, lw=0, label=label)
    left = left + volume
ax_vol.set_facecolor("0.8")  # grey: samples the solver could not constrain
ax_gr.set(xlabel="GR (API)", ylabel="Depth (m)")
ax_vol.set(xlim=(0, 1), xlabel="Volume (v/v)")
ax_gr.invert_yaxis()
ax_vol.legend(handles=ax_vol.get_legend_handles_labels()[0]
              + [plt.Rectangle((0, 0), 1, 1, color="0.8", label="unsolved")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=7,
              handlelength=1, columnspacing=0.8)
fig.suptitle("15/9-14 chalk: mineral volumes")
thumb("cap_minerals", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: Pickett plot and Sw models"
  code_prompt_hide: "Hide the code: Pickett plot and Sw models"
---
from petrophys.visualization import create_bar_comparison, create_pickett_plot

pickett = create_pickett_plot(
    interpreted, rw=0.07, rt_curve="RDEP", color_curve="VCL",
    depth_range=(3223, 3556), a=0.8, m=2.0, n=2.0, figsize=THUMB,  # Statfjord water leg
)
thumb("cap_pickett", pickett.fig)

net_pay = {}
for model in ["archie", "simandoux", "indonesia", "dual_water", "auto"]:
    run = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model=model)
    net_pay[model] = run.summary(top=BRENT[0], bottom=BRENT[1])["net_thickness"]["net_pay"]
print("Brent net pay (m):")
for model, pay in net_pay.items():
    print(f"  {model:<11}{pay:6.1f}")
fig, ax = create_bar_comparison(list(net_pay), list(net_pay.values()), value_labels=True,
                                ylabel="Brent net pay (m)", title="Net pay by Sw model, 30/6-5",
                                figsize=THUMB)
ax.set_ylim(0, 1.2 * max(net_pay.values()))
thumb("cap_sw", fig)
```

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: core, flow units and formation summary"
  code_prompt_hide: "Hide the code: core, flow units and formation summary"
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

summary = result.formation_summary(rank="formation")
columns = {"Formation": "Formation", "Net Pay (m)": "Net pay (m)",
           "Avg PHIE": "PHIE", "Avg Sw": "Sw"}
brent_down = summary.loc[summary["Top"] >= BRENT[0], list(columns)].rename(columns=columns)
table("cap_formations", brent_down)
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

# Saturation-height: fit a J-function on HW-3 (Kharaib-2) below an assumed FWL.
hw3 = pp.read(pp.sample("HW-3")).to_dataframe()
phi, perm, sw = hw3["PHIE_CC"].values, hw3["PERM_CC"].values, hw3["SW_INPUT"].values
fwl = hw3["DEPTH"][np.isfinite(phi * perm * sw)].max() + 20.0  # ft, assumed FWL
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

```{code-cell} python
---
tags: [hide-input]
mystnb:
  code_prompt_show: "Show the code: multi-well, trace, context and MCP"
  code_prompt_hide: "Hide the code: multi-well, trace, context and MCP"
---
from petrophys.workflow import BatchProcessor

# Seventeen COSTA wells, each covering the Kharaib-2 interval only.
costa_tops = pp.sample("formations_costa")
wells = [pp.read(pp.sample(name), formations=costa_tops) for name in pp.list_samples("costa")]
batch = BatchProcessor(config=dict(vcl_cutoff=0.30, phie_cutoff=0.10, sw_cutoff=0.60))
runs = batch.process_wells(wells)
pay = {run.well_name: run.result.formation_summary().iloc[0]["Net Pay (ft)"] for run in runs}
fig, ax = create_bar_comparison(list(pay), list(pay.values()), sort_by=0, figsize=THUMB,
                                ylabel="Net pay (ft)", title="Kharaib-2 net pay, COSTA field")
thumb("cap_costa", fig)

steps = result.to_dataframe()
steps = steps[steps["Step"].isin(["vclay", "porosity", "lithology", "saturation",
                                  "permeability", "net_pay"])]
table("cap_trace", steps[["Step", "Method"]])

from mcp import Client
from petrophys.mcp import create_server

context = pp.build_context(result)
async with Client(create_server(roots=["."])) as client:
    tools = [tool.name for tool in (await client.list_tools()).tools]
table("cap_llm", pd.DataFrame({
    "Output": ["Context JSON", "MCP server"],
    "Contents": [
        f"{len(json.dumps(context, default=str)) // 1000} kB; sections: "
        + ", ".join(k for k in context if isinstance(context[k], dict)),
        f"{len(tools)} tools: " + ", ".join(tools[:6]) + ", ...",
    ],
}))
```

## Go deeper

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Complete tour
:link: tutorials/complete_tour
:link-type: doc
One notebook through the whole library on real wells.
:::

:::{grid-item-card} Case study: 30/6-5
:link: tutorials/case_studies/cs1_full_formation_evaluation_30_6-5
:link-type: doc
A full formation evaluation of the Brent reservoir.
:::

:::{grid-item-card} Case study: COSTA field
:link: tutorials/case_studies/cs3_field_screening_costa_17wells
:link-type: doc
Screening 17 carbonate wells for pay.
:::

:::{grid-item-card} Specialist data
:link: specialist_data
:link-type: doc
NMR, spectral gamma, core and saturation-height in detail.
:::
::::
