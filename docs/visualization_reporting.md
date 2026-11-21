---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
  execution_in_temp: true
---

# Visualization and reporting

Whether `pp.interpret` or your own workflow produced it, an interpretation is delivered as plots and files.
Every plot takes a `Well` or an interpretation result and finds curves by their standard names; tracks without data are left out.
This page shows each plot type on the bundled wells, then every export format.

```{code-cell} python
:tags: [remove-cell]

import warnings

# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The page explains this once, in the text.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

The examples interpret 30/6-5 once, with Rw 0.07 ohm.m at 230 °F. Its core SWC and SOC are percent values labelled as fractions. Petrophys converts them and warns; this page hides that warning.

```{code-cell} python
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"),
               lithology=pp.sample("lithology_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
pp.plot(result, top=2800, bottom=3000)
```

## Figure style and export

Library plots share DejaVu Sans typography, white backgrounds and PNG export
defaults. Curve colours, lithology patterns and specialist colour scales retain
their scientific meaning. Importing the library does not change Matplotlib's
global settings.

For custom Matplotlib figures in the same notebook, call
`petrophys.visualization.use_plot_style()` once. For temporary overrides, use
`with petrophys.visualization.plot_style({"font.size": 10}):` around your plotting
code. Existing axes and explicit curve styling remain under your control.

Save any figure with `petrophys.visualization.save_figure(fig, "plot.png")`.
Exports use 150 dpi, a white background and tight margins; pass `dpi=300` for
higher-resolution raster output, or use a `.pdf` or `.svg` extension for vectors.
Viewer `.save(...)` methods use the same defaults.

## Log plot templates

`pp.plot` builds its tracks from the curves present (`template="auto"`, above).
A named template gives a fixed layout instead: `raw`, `triple_combo`, `petrophysics`,
`interpretation`, `minerals`, `comprehensive` and a few more. The examples below show
the Brent Group (2841-2958 m) with a margin.

The `petrophysics` template keeps the interpretation tracks:

```{code-cell} python
pp.plot(result, top=2830, bottom=2970, template="petrophysics")
```

`comprehensive` adds the mineral volumes of the multi-mineral solution and a seal flag:

```{code-cell} python
pp.plot(result, top=2830, bottom=2970, template="comprehensive")
```

`triple_combo` is the classic first look at gamma ray, resistivity and density-neutron
({func}`~petrophys.visualization.create_triple_combo` draws the same):

```{code-cell} python
pp.plot(result, top=2830, bottom=2970, template="triple_combo")
```

A well plots before any interpretation; only the measured curves are drawn:

```{code-cell} python
pp.plot(well, top=2830, bottom=2970)
```

## Crossplots

Crossplots convert declared compatible units before plotting and leave the source well unchanged. Unknown or incompatible units raise a named error; missing units warn before assuming the chart unit. Pickett, Hingle and Buckles accept `xlim=` and `ylim=`; a note counts finite points outside the displayed range.

Each crossplot answers one question. The lithology plots use the interval where PEF
and sonic are both logged (2784-3176 m); the resistivity plots use the Statfjord water
leg (3223-3556 m), where the clean sands plot near the water line (Sw 80-100 %).

**Neutron-density** shows lithology and gas against the sandstone, limestone and
dolomite lines. The neutron-density and M-N references assume freshwater and
limestone-referenced neutron response with linear mineral endpoints; they are not
vendor-specific tool calibration charts:

```{code-cell} python
from petrophys.visualization import (
    create_buckles_plot, create_hingle_plot, create_mid_plot,
    create_mn_plot, create_nd_crossplot, create_pickett_plot,
)

create_nd_crossplot(result, depth_range=(2784, 3176));
```

**M-N plot**: matrix minerals from density, neutron and sonic together:

```{code-cell} python
create_mn_plot(result, depth_range=(2784, 3176));
```

**MID plot**: apparent matrix density against apparent matrix slowness. It removes
the freshwater contribution using `PHIT`, or an explicit total-porosity curve
selected with `porosity_curve="MY_PHIT"`. Without total porosity it estimates
porosity from linear quartz/calcite/water and calcite/dolomite/water triangles.
Samples outside that clean binary-mineral model warn and are omitted. For gas,
clay or complex mineral mixtures, supply independently determined total porosity:

```{code-cell} python
create_mid_plot(result, depth_range=(2784, 3176));
```

**Pickett plot**: log Rt against log porosity. Water-bearing rock falls on a straight
line whose slope is -m and whose intercept at porosity 1 is a * Rw; with a known,
Rw can be recovered. Rw must be at formation temperature. Lines of constant Sw lie
above it. Here with the operator's Archie constants (a 0.8, m and n 2):

```{code-cell} python
create_pickett_plot(result, rw=0.07, a=0.8, rt_curve="RDEP", color_curve="VCL",
                    depth_range=(3223, 3556));
```

**Hingle plot**: the same Archie water line on linear axes, where it passes through
the matrix point at zero porosity:

```{code-cell} python
create_hingle_plot(result, rw=0.07, a=0.8, depth_range=(3223, 3556));
```

**Buckles plot**: each hyperbola represents constant bulk-volume water (porosity * Sw).
Clustering along one hyperbola can support an irreducible-water interpretation,
but constant BVW alone does not establish irreducible saturation. Only the
physical segment with Sw <= 1 is drawn. The axis identifies total porosity when
PHIT supplies the fallback. HW-3,
a COSTA carbonate whose header carries Rw:

```{code-cell} python
create_buckles_plot(pp.interpret(pp.sample("HW-3")));
```

## Reservoir displays

Flow units, net flags and formation tops sum up where the good rock is.

**Flow units** plot the reservoir quality index (RQI) against normalized porosity
(φz) for the core plugs; each hydraulic flow unit lies on its own line:

```{code-cell} python
from petrophys.interpretation.core_analysis import core_data_from_well
from petrophys.visualization import create_flow_unit_plot, plot_zone_flags

create_flow_unit_plot(core_data_from_well(result));
```

**Zone flags** draw net rock, net reservoir and net pay as depth bars:

```{code-cell} python
plot_zone_flags(result, flag_curves=["NET_ROCK", "NET_RES", "NET_PAY"],
                depth_range=(2841, 2958));
```

Formation tops given to `pp.read` (or to `pp.plot` as `formations=`) are drawn on
every log plot, as above. To shade named intervals on a Matplotlib axis of your own,
use {func}`~petrophys.visualization.add_zone_shading`.

## Multi-well and sensitivity charts

`petrophys.visualization` also draws charts that are not depth tracks.
`create_bar_comparison` and `create_line_comparison` complete the set.

**Multi-well overlay**: one curve from several wells on a shared depth axis, here the
normalized gamma ray of five COSTA wells:

```{code-cell} python
from petrophys.visualization import (
    create_heatmap, create_histogram_comparison, create_multiwell_overlay,
    create_tornado_plot,
)

costa = pp.read([pp.sample(name) for name in ["HW-3", "HW-4", "HW-5", "HW-6", "HW-7"]])
create_multiwell_overlay(costa, "GR", normalize=True, title="Normalized GR, five COSTA wells");
```

**Histogram comparison**: distributions on common bins after conversion to a
shared unit. Equivalent percent and fraction porosities share a distribution;
incompatible declared units raise an error. For example:

```{code-cell} python
create_histogram_comparison(costa, "GR", density=True, xlabel="GR (API)", title="GR by well");
```

**Tornado**: how far each assumption moves the answer.
{func}`~petrophys.evaluation.cutoff_sensitivity` re-applies the net-pay cutoffs to
the interpreted curves, which stay fixed; it applies the cutoffs alone, so the samples
`pp.interpret` leaves out of `NET_PAY` (bad hole, evaporite, coal, halite) count
here. The standard clastic cutoffs are VCL 0.4, PHIE 0.08 and Sw 0.6. Each case
moves one cutoff to a low and a high value:

```{code-cell} python
from petrophys.evaluation import cutoff_sensitivity

def net_pay(**cutoffs):
    grid = cutoff_sensitivity(result.well, top=2841, bottom=2958, **cutoffs)
    return grid["net_pay_thickness"].tolist()

low, high = zip(net_pay(vcl_cutoffs=[0.3, 0.5]), net_pay(phie_cutoffs=[0.12, 0.05]),
                net_pay(sw_cutoffs=[0.5, 0.7]))
create_tornado_plot(["VCL cutoff 0.3 / 0.5", "PHIE cutoff 0.12 / 0.05", "Sw cutoff 0.5 / 0.7"],
                    low, high, base=net_pay()[0], xlabel="Brent Group net pay (m)");
```

With the standard cutoffs the Brent Group has 23.3 m of net pay. The Sw cutoff moves
it most, from 21.0 to 25.6 m; the VCL and PHIE cutoffs move it by less than 1.3 m
either way.

**Heatmap**: any 2-D grid, such as net-to-gross over two cutoffs at once:

```{code-cell} python
grid = cutoff_sensitivity(result.well, top=2841, bottom=2958,
                          vcl_cutoffs=[0.2, 0.3, 0.4, 0.5], sw_cutoffs=[0.4, 0.5, 0.6, 0.7, 0.8])
ntg = grid.pivot(index="vcl_cutoff", columns="sw_cutoff", values="ntg_pay")
create_heatmap(ntg, x=ntg.columns, y=ntg.index, cmap="RdYlGn", invert_yaxis=False,
               xlabel="Sw cutoff", ylabel="VCL cutoff", colorbar_label="Net-to-gross pay",
               title="Brent Group net-to-gross pay", figsize=(7, 4.5));
```

Specialist displays (NMR panel, borehole images, XRF, saturation-height, core plots) are shown in {doc}`specialist_data`.
The XRF panel converts all elements to one mass-concentration unit (`concentration_unit="%"`
by default, or `"ppm"`) before drawing shared axes and calculating proxies.
Missing element units warn and assume percent; unknown units raise an error.

## Controlling the log viewer

`pp.plot` takes a well, a result or a well file (`pp.plot("well.las")`) and returns a {class}`~petrophys.visualization.LogViewer`, a matplotlib figure shown inline in a notebook and saved by `.save(path)` as PNG, PDF, SVG or any other matplotlib format. Choose the window and tracks, then read what it did.
Several wells give a {class}`~petrophys.visualization.MultiWellPlot` (each well's own plot, one at a time; `.save(folder)` writes a file per well), or with `curves=` a {class}`~petrophys.visualization.WellComparisonPlot` (one figure, a track per curve with every well in it).

The `auto` template of the interpreted 30/6-5 has the tracks DEPTH, LITHOLOGY,
GAMMA / HOLE, RESISTIVITY, DENSITY / NEUTRON, SONIC, POROSITY, SATURATION,
PERMEABILITY, ROCK QUALITY, MINERAL VOLUMES and FLAGS; `tracks=` keeps a few of them:

```{code-cell} python
focus = pp.plot(result, top=2841, bottom=2958,
                tracks=["GAMMA / HOLE", "POROSITY", "SATURATION", "FLAGS"])
focus
```

The viewer reports what it did. The Dunlin Group and Drake Formation tops sit at the
bottom of this window, so their labels are suppressed:

```{code-cell} python
focus.diagnostics["suppressed_formation_labels"]
```

| Control or attribute | What it does |
| --- | --- |
| `top=`, `bottom=` | Window in the well's depth unit; each defaults to its end of the logged interval |
| `tracks=[...]` | Track display names to keep, in order; also draws the optional TEMPERATURE and RW / RMF tracks |
| `curves=[...]` | Only these curves, each in its usual track, by the file's names or standard names (`GR`, `RDEP`); a curve no track takes gets a generic track. Not together with `tracks=` |
| `curve_styles={"GR": {"color": "black", "right": 200}}` | Colour, scale, line, fill or marker of single curves, without an XML template |
| `curve_aliases=` | Your mnemonics to standard names: a dict, or a CSV/Excel table with `curve` and `standard` columns |
| `template=` / `template_xml_path=` | Named layout, or an XML file with your own tracks, scales and fills |
| `formations=`, `lithology=` | Formation tops and lithology intervals for this plot only, as for `pp.read`; geology passed to `pp.read` also feeds the interpretation |
| `formation_name=` | Formation name shown in the plot title |
| `lithology_codes=` | Codebook of a lithology code curve: a mapping of code to rock name, or a CSV/Excel codebook |
| `strict=True` | Fail when a fixed template needs a missing curve, instead of skipping the track |
| `viewer.omitted_tracks` | Tracks left out because their curves are missing |
| `viewer.diagnostics` | Imports, curve choices, omitted empty tracks, optional tracks left out, and labels suppressed to avoid crowding |
| `viewer.warnings` | Plot-only warnings, such as a curve whose values run off its track scale |
| `viewer.save(path)`, `.show()`, `.close()` | Output |

A style or window you give is drawn as given. For example, 100 m from the
Brent Group top, the lithology fill at 30 % opacity, and your own lithology codes:

```python
pp.plot(result, top=2841, bottom=2941,
        curve_styles={"LITHOLOGY_GEOLINK": {"fill_alpha": 0.3}})
pp.plot(result, lithology_codes={65000: "Sandstone", 30000: "Shale"})
```

### Your curve names, your curves, your styles

The Teapot Dome well 44-MDX-10 uses company mnemonics (`DT`, `SPR`, `LLS`, `LLD`,
`MSFL`, `NPOR`, `RHOB`, `CORR`, `GRD`, `CALD`). Standard names find most of them;
its spontaneous potential `SPR` is known to no alias table, so a mapping places it.
The mapping can be a dict or a CSV/Excel table (`curve`, `standard`), given to
`pp.read` (stored on the well, used by plotting and interpretation alike) or to one
`pp.plot`. `curve_styles` restyles single curves:

```{code-cell} python
pp.plot(pp.sample("teapot_44-mdx-10"), curves=["GR", "SP", "RDEP", "RMED", "NPHI", "RHOB"],
        curve_aliases={"SPR": "SP"}, top=2400, bottom=3000,
        curve_styles={"GR": {"color": "black"}, "SP": {"color": "#984ea3"}})
```

Coincident formation tops are grouped, not discarded. Plotting never computes a missing curve: a well without Sw shows no Sw. {func}`~petrophys.visualization.create_adaptive_log_viewer_template` lists the tracks `auto` would draw before you render.

## Exports

Write curves for other software, a full trace for review, or a delivery package for a folder of wells.
The interpretation writes its own LAS file and trace:

```{code-cell} python
result.to_las("30_6-5_interpreted.las")   # curves + settings in ~Params
result.to_json("30_6-5_trace.json")       # every step, input and decision
result.to_markdown("30_6-5_trace.md");
```

Chosen curves go to CSV or Excel, and a delivery LAS standardizes the curve families:

```{code-cell} python
from petrophys.io import export_csv, export_excel
from petrophys.reporting import export_example_format_las

curves = ["GR", "VCL", "PHIE", "SW", "PERM", "NET_PAY"]
export_csv(result.well, "30_6-5.csv", curves=curves)
export_excel(result.well, "30_6-5.xlsx", curves=curves)
export_example_format_las(result.well, "delivery/30_6-5.las", set_name="Petrophys interpretation");
```

Several wells interpret together and write a LAS file and a plot per well:

```{code-cell} python
results = pp.interpret(costa, lithology="carbonate", carbonate_calibration="costa")
results.to_las("interpreted")
results.save_plots("plots");
```

```{code-cell} python
from pathlib import Path

sorted(path.as_posix() for path in Path().rglob("*.*"))
```

| Export | Call | Contents |
| --- | --- | --- |
| Interpreted LAS | `result.to_las(path)`, or `pp.interpret(well, output=path)` | All curves with units and descriptions; settings as `PP_*` parameters; executed and skipped steps in `~Other` |
| Trace | `result.to_json(path)`, `result.to_markdown(path)` | Every step's method, inputs, outputs, parameters and decisions |
| Tables | `export_csv`, `export_excel` from `petrophys.io` | Chosen curves with depth; missing values left blank |
| Delivery LAS | `export_example_format_las` | Standardized curve families with calculation provenance |
| Several wells | `results.to_las(folder)`, `results.save_plots(folder)` | One interpreted LAS and one plot per well |
| Work package | `export_example_work_package_directory` | Runs the workflow on each LAS in a folder: interpreted LAS plus reference JSON |
| Batch plots | `export_interpretation_plots`, `write_visualization_report` | One PNG per LAS in a folder and a CSV of what was drawn or skipped |

For LLM-ready JSON grouped by formation and lithology, see {doc}`context`.

## Next

{doc}`specialist_data`: NMR, spectral gamma, core, saturation-height and borehole-image data, with their own displays.

{doc}`Tutorial 17 <tutorials/part_b/17_visualization_masterclass>` shows every plot type with the reasoning behind it, and {doc}`tutorial 18 <tutorials/part_b/18_multiwell_field_study>` adds batch runs, normalization and field charts.
