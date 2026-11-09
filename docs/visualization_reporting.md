---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Visualization and reporting

Whether `pp.interpret` or your own workflow produced it, an interpretation is delivered as plots and files.
Every plot takes a `Well` or an interpretation result and finds curves by their standard names; tracks without data are left out.
This page shows each plot type on the bundled wells, then every export format.

```{code-cell} python
:tags: [remove-cell]

%matplotlib inline
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter
from myst_nb import glue

# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The page explains this once, in the text.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
pd.set_option("display.max_colwidth", None)
from petrophys.visualization import use_plot_style
use_plot_style()


def place(key, fig):
    """Put a figure in the tab or card that names ``key``."""
    for ax in fig.axes:  # small log axes: keep only the decade labels
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
    glue(key, fig, display=False)
    plt.close(fig)
```

The examples interpret 30/6-5 once, with Rw 0.07 ohm.m at 230 °F. Its core SWC and SOC are percent values labelled as fractions. Petrophys converts them and warns; this page hides that warning.

```{code-cell} python
import numpy as np
import pandas as pd
import petrophys as pp

BRENT = (2841, 2958)  # Brent Group reservoir, m
well = pp.read(
    pp.sample("30_6-5"),
    formations=pp.sample("formations_30_6-5"),
    lithology=pp.sample("lithology_30_6-5"),
)
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)

viewer = pp.plot(result, top=2800, bottom=3000, figsize=(15, 11))
viewer.show()
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

`pp.plot` builds its tracks from the curves present (`template="auto"`, above). Named templates give a fixed layout, {func}`~petrophys.visualization.create_triple_combo` draws the classic first look, and a well plots before any interpretation. Below, `place()` is a small page helper that puts each figure in its tab or card.

```{code-cell} python
from petrophys.visualization import DEFAULT_LOG_VIEWER_TEMPLATES, create_triple_combo

from textwrap import fill

TOP, BOTTOM = 2830, 2970  # the Brent Group with a margin
print(fill("Named templates: " + ", ".join(DEFAULT_LOG_VIEWER_TEMPLATES), 88))
for name, size in [("petrophysics", (14, 10.5)), ("comprehensive", (18, 12))]:
    place(name, pp.plot(result, top=TOP, bottom=BOTTOM, template=name, figsize=size).fig)
triple = create_triple_combo(result.well, depth_range=(TOP, BOTTOM), figsize=(10, 10))
place("triple_combo", triple.fig)
raw = pp.plot(well, top=TOP, bottom=BOTTOM, figsize=(13, 10))  # no pp.interpret needed
place("uninterpreted", raw.fig)
```

::::{tab-set}
:::{tab-item} petrophysics
```{glue:figure} petrophysics
```
:::
:::{tab-item} comprehensive
```{glue:figure} comprehensive
```
:::
:::{tab-item} triple combo
```{glue:figure} triple_combo
```
:::
:::{tab-item} before interpretation
```{glue:figure} uninterpreted
```
:::
::::

## Crossplots

Each crossplot answers one question. The lithology plots use the interval where PEF and sonic are logged; the resistivity plots use the Statfjord water leg.

```{code-cell} python
from petrophys.visualization import (
    create_buckles_plot, create_hingle_plot, create_mid_plot,
    create_mn_plot, create_nd_crossplot, create_pickett_plot,
)

geo = result.well
LITH = (2784, 3176)   # PEF and sonic both logged
WATER = (3223, 3556)  # Statfjord water leg
SMALL = (5, 4.2)
carbonate = pp.interpret(pp.read(pp.sample("HW-3")))  # COSTA carbonate, Rw in header

place("nd", create_nd_crossplot(geo, depth_range=LITH, figsize=SMALL))
place("mn", create_mn_plot(geo, depth_range=LITH, figsize=SMALL))
place("mid", create_mid_plot(geo, depth_range=LITH, figsize=SMALL))
place("pickett", create_pickett_plot(geo, rw=0.07, rt_curve="RDEP", color_curve="VCL",
                                     a=0.8, depth_range=WATER, figsize=SMALL).fig)
place("hingle", create_hingle_plot(geo, rw=0.07, a=0.8, depth_range=WATER, figsize=SMALL))
place("buckles", create_buckles_plot(carbonate.well, figsize=SMALL))
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Neutron-density
```{glue:figure} nd
```
Lithology and gas: sandstone, limestone and dolomite lines. `create_nd_crossplot`
:::
:::{grid-item-card} M-N plot
```{glue:figure} mn
```
Matrix minerals from density, neutron and sonic. `create_mn_plot`
:::
:::{grid-item-card} MID plot
```{glue:figure} mid
```
Apparent matrix density against apparent matrix slowness. `create_mid_plot`
:::
:::{grid-item-card} Pickett plot
```{glue:figure} pickett
```
Rw and m from the water line; Sw lines above it. `create_pickett_plot`
:::
:::{grid-item-card} Hingle plot
```{glue:figure} hingle
```
The Archie water line on linear axes. `create_hingle_plot`
:::
:::{grid-item-card} Buckles plot (HW-3)
```{glue:figure} buckles
```
Points on one PHIE × Sw hyperbola are at irreducible water. `create_buckles_plot`
:::
::::

## Reservoir displays

Flow units, net flags and formation bands sum up where the good rock is.

```{code-cell} python
from petrophys.evaluation import calculate_flow_units
from petrophys.visualization import (
    LogPlot, add_zone_shading, create_flow_unit_plot, plot_zone_flags,
)

plugs = np.isfinite(geo["PHIC"].data) & np.isfinite(geo["KH_CORE"].data)
# Permeability first, then porosity:
units = calculate_flow_units(geo["KH_CORE"].data[plugs], geo["PHIC"].data[plugs])
place("flow_units", create_flow_unit_plot(result=units, figsize=SMALL))

fig, ax = plot_zone_flags(geo, flag_curves=["NET_ROCK", "NET_RES", "NET_PAY"],
                          depth_range=BRENT, figsize=SMALL)
place("zone_flags", fig)

tops = pp.read_formations(pp.sample("formations_30_6-5"))
brent_formations = tops[(tops["rank"] == "formation") & (tops["top_depth"] >= BRENT[0])
                        & (tops["bottom_depth"] <= BRENT[1])]
panel = LogPlot(geo, figsize=SMALL)
panel.create_figure(n_tracks=2)
panel.add_curve(0, "GR", color="#1b7837", xlim=(0, 150))
panel.add_curve(1, "PHIE", color="navy", xlim=(0.4, 0))
panel.set_depth_range(top=BRENT[0] - 10, bottom=BRENT[1] + 10)
colors = ["#fee08b", "#9ecae1"]
zones = [{"top": row.top_depth, "bottom": row.bottom_depth, "name": row.formation_name,
          "color": color} for row, color in zip(brent_formations.itertuples(), colors)]
for axis in panel.axes:
    add_zone_shading(axis, zones, alpha=0.35)
place("shading", panel.fig)
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Flow units
```{glue:figure} flow_units
```
RQI against φz for the core plugs, one line per unit. `create_flow_unit_plot`
:::
:::{grid-item-card} Zone flags
```{glue:figure} zone_flags
```
Net rock, reservoir and pay as depth bars. `plot_zone_flags`
:::
:::{grid-item-card} Formation shading
```{glue:figure} shading
```
Named bands on any matplotlib axis. `add_zone_shading`
:::
::::

## Multi-well and sensitivity charts

`petrophys.visualization` also draws charts that are not depth tracks, from plain arrays or `Well` objects. `create_bar_comparison` and `create_line_comparison` complete the set.

```{code-cell} python
from petrophys.evaluation import STANDARD_CUTOFFS, NetPayModule
from petrophys.visualization import (
    create_heatmap, create_histogram_comparison, create_multiwell_overlay,
    create_tornado_plot,
)

costa = [pp.read(pp.sample(name)) for name in ["HW-3", "HW-4", "HW-5", "HW-6", "HW-7"]]
fig, ax = create_multiwell_overlay(costa, "GR", normalize=True, figsize=SMALL,
                                   title="Normalized GR, five COSTA wells")
place("overlay", fig)
gr_by_well = {w.name: w["GR"].data for w in costa}
fig, ax = create_histogram_comparison(gr_by_well, density=True, xlabel="GR (API)",
                                      title="GR by well", figsize=SMALL)
place("histogram", fig)

# Brent Group net pay as the cutoffs move; the interpreted curves stay fixed. NetPayModule
# applies the cutoffs alone, so the samples pp.interpret leaves out of NET_PAY
# (bad hole, evaporite, coal, halite) count here.
brent = geo.depth_slice(*BRENT)
base = STANDARD_CUTOFFS["clastic"]
def net_pay(**change):
    return NetPayModule(**{**base, **change}).calculate(brent).net_pay_thickness

cases = {"VCL cutoff 0.3 / 0.5": ("vcl_cutoff", 0.3, 0.5),
         "PHIE cutoff 0.12 / 0.05": ("phie_cutoff", 0.12, 0.05),
         "Sw cutoff 0.5 / 0.7": ("sw_cutoff", 0.5, 0.7)}
low = [net_pay(**{key: lo}) for key, lo, hi in cases.values()]
high = [net_pay(**{key: hi}) for key, lo, hi in cases.values()]
fig, ax = create_tornado_plot(list(cases), low, high, base=net_pay(),
                              xlabel="Brent Group net pay (m)",
                              title="Net pay sensitivity to cutoffs", figsize=SMALL)
place("tornado", fig)

vcl_cuts, sw_cuts = np.arange(0.2, 0.55, 0.05), np.arange(0.4, 0.85, 0.05)
phie_cut = base["phie_cutoff"]
ntg = [[NetPayModule(vcl_cutoff=v, phie_cutoff=phie_cut, sw_cutoff=s).calculate(brent)
        for s in sw_cuts] for v in vcl_cuts]
ntg = np.array([[r.net_pay_thickness / r.gross_thickness for r in row] for row in ntg])
fig, ax = create_heatmap(ntg, x=sw_cuts, y=vcl_cuts, cmap="RdYlGn", invert_yaxis=False,
                         xlabel="Sw cutoff", ylabel="VCL cutoff",
                         colorbar_label="Net-to-gross",
                         title="Brent Group net-to-gross", figsize=SMALL)
place("heatmap", fig)
print("Standard clastic cutoffs:", base)
print(f"Brent Group net pay with them, cutoffs alone: {net_pay():.1f} m")
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Multi-well overlay
```{glue:figure} overlay
```
One curve from several wells on a shared depth axis. `create_multiwell_overlay`
:::
:::{grid-item-card} Histogram comparison
```{glue:figure} histogram
```
Distributions on common bins, e.g. before normalization. `create_histogram_comparison`
:::
:::{grid-item-card} Tornado
```{glue:figure} tornado
```
How far each assumption moves the answer. `create_tornado_plot`
:::
:::{grid-item-card} Heatmap
```{glue:figure} heatmap
```
Any 2-D grid, such as net-to-gross over two cutoffs. `create_heatmap`
:::
::::

Specialist displays (NMR panel, borehole images, XRF, saturation-height, core plots) are shown in {doc}`specialist_data`.

## Controlling the log viewer

`pp.plot` takes a well, a result or a well file (`pp.plot("well.las")`) and returns a {class}`~petrophys.visualization.LogViewer`. Choose the window and tracks, then read what it did.

```{code-cell} python
tracks = [track.display_name for track in viewer.template.tracks]
print(fill("Track names: " + ", ".join(tracks), 88, subsequent_indent="    "))

focus = pp.plot(result, top=BRENT[0], bottom=BRENT[1], figsize=(9, 9.5),
                tracks=["GAMMA / HOLE", "POROSITY", "SATURATION", "FLAGS"])
print("Omitted tracks:", focus.omitted_tracks)
print("Lithology from:", focus.diagnostics["lithology_source"])
print("Labels suppressed:", focus.diagnostics["suppressed_formation_labels"])
warning = f"Warnings: {len(focus.warnings)}; the first: {focus.warnings[0]}"
print(fill(warning, 88, subsequent_indent="    "))
focus.show()
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
| `lithology_codes=` | Codebook of a lithology code curve: a mapping of code to rock name, or a CSV/Excel codebook |
| `strict=True` | Fail when a fixed template needs a missing curve, instead of skipping the track |
| `viewer.omitted_tracks` | Tracks left out because their curves are missing |
| `viewer.diagnostics` | Imports, curve choices, omitted empty tracks, optional tracks left out, and labels suppressed to avoid crowding |
| `viewer.warnings` | Plot-only warnings |
| `viewer.save(path)`, `.show()`, `.close()` | Output |

A style or window you give is drawn as given. For example, 100 m from the
Brent Group top, the lithology fill at 30 % opacity, and your own lithology codes:

```python
pp.plot(result, top=2841, bottom=2941,
        curve_styles={"LITHOLOGY_GEOLINK": {"fill_alpha": 0.3}})
pp.plot(result, lithology_codes={65000: "Sandstone", 30000: "Shale"})
```

### Your curve names, your curves, your styles

The Teapot Dome well 44-MDX-10 uses company mnemonics. Standard names find most of
them; its spontaneous potential `SPR` is known to no alias table, so a mapping
places it. The mapping can be a dict or a CSV/Excel table (`curve`, `standard`),
given to `pp.read` (stored on the well, used by plotting and interpretation alike)
or to one `pp.plot`.

```{code-cell} python
teapot = pp.read(pp.sample("teapot_44-mdx-10"))
print("File curves:", list(teapot.curves))
own = pp.plot(teapot, curves=["GR", "SP", "RDEP", "RMED", "NPHI", "RHOB"],
              curve_aliases={"SPR": "SP"}, top=2400, bottom=3000, figsize=(9, 8),
              curve_styles={"GR": {"color": "black"}, "SP": {"color": "#984ea3", "line_width": 1.2}})
print("Drawn:", own.diagnostics["curve_lookup"])
own.show()
```

Coincident formation tops are grouped, not discarded. Plotting never computes a missing curve: a well without Sw shows no Sw. {func}`~petrophys.visualization.create_adaptive_log_viewer_template` lists the tracks `auto` would draw before you render.

## Exports

Write curves for other software, a full trace for review, or a delivery package for a folder of wells.

```{code-cell} python
import shutil
import tempfile
from pathlib import Path

from petrophys.io import export_csv, export_excel
from petrophys.reporting import (
    export_example_format_las, export_example_work_package_directory,
)
from petrophys.visualization import (
    export_interpretation_plots, write_visualization_report,
)

curves = ["GR", "VCL", "PHIE", "SW", "PERM", "NET_PAY"]
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    result.to_las(out / "30_6-5_interpreted.las")        # curves + settings in ~Params
    result.to_json(out / "30_6-5_trace.json")            # every step, input and decision
    result.to_markdown(out / "30_6-5_trace.md")
    export_csv(result.well, out / "30_6-5.csv", curves=curves)
    export_excel(result.well, out / "30_6-5.xlsx", curves=curves)
    export_example_format_las(result.well, out / "delivery" / "30_6-5.las",
                              set_name="Petrophys interpretation")

    logs = out / "logs"                                  # a folder of LAS files
    logs.mkdir()
    shutil.copy(pp.sample("30_6-5"), logs)
    export_example_work_package_directory(logs, out / "package")
    plots = export_interpretation_plots(logs, out / "plots", figsize=(10, 7), dpi=60)
    write_visualization_report(plots, out / "plots" / "plots.csv")

    for path in sorted(out.rglob("*.*")):
        if path.parent != logs:
            size = path.stat().st_size / 1e3
            print(f"{path.relative_to(out).as_posix():28s} {size:8,.0f} kB")
```

| Export | Call | Contents |
| --- | --- | --- |
| Interpreted LAS | `result.to_las(path)`, or `pp.interpret(well, output=path)` | All curves with units and descriptions; settings as `PP_*` parameters; executed and skipped steps in `~Other` |
| Trace | `result.to_json(path)`, `result.to_markdown(path)` | Every step's method, inputs, outputs, parameters and decisions |
| Tables | `export_csv`, `export_excel` from `petrophys.io` | Chosen curves with depth; missing values left blank |
| Delivery LAS | `export_example_format_las` | Standardized curve families with calculation provenance |
| Work package | `export_example_work_package_directory` | Runs the workflow on each LAS in a folder: interpreted LAS plus reference JSON |
| Batch plots | `export_interpretation_plots`, `write_visualization_report` | One PNG per LAS and a CSV of what was drawn or skipped |

For LLM-ready JSON grouped by formation and lithology, see {doc}`context`.

## Next

{doc}`specialist_data`: NMR, spectral gamma, core, saturation-height and borehole-image data, with their own displays.

{doc}`Tutorial 17 <tutorials/part_b/17_visualization_masterclass>` shows every plot type with the reasoning behind it, and {doc}`tutorial 18 <tutorials/part_b/18_multiwell_field_study>` adds batch runs, normalization and field charts.
