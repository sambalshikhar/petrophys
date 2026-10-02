---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Data and depth

A {class}`~petrophys.model.Well` holds one depth index, a set of
{class}`~petrophys.model.Curve` objects, header values and any attached
geology. This page takes the bundled 30/6-5 well apart.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import textwrap
import warnings
import pandas as pd

# Display settings for this page: whole table cells, warnings wrapped and without file paths.
pd.set_option("display.max_colwidth", None)
warnings.formatwarning = lambda message, category, *args, **kwargs: textwrap.fill(
    f"{category.__name__}: {message}", 88, subsequent_indent="    ") + "\n"
```

```{code-cell} python
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))
summary = well.describe()
for key, value in summary.items():
    if key != "curves":
        print(f"{key:<20}{round(value, 3) if isinstance(value, float) else value}")
```

## The well at a glance

| To get | Use |
| --- | --- |
| Curve names | `well.curves` |
| One curve | `well["GR"]`, or `well.get("GR")`, which returns `None` when absent |
| Depths and their unit | `well.depth` (a NumPy array), `well.depth_unit` |
| Header values | `well.get_parameter("BHT")`, or `well.describe()["header_parameters"]` |
| A depth window | `well.depth_slice(top, bottom)`, a new well |
| A table | `well.to_dataframe(curves=[...])` |
| A new curve | `well.add_curve(name, data, unit, description)` |
| Files | `well.to_las(path)`, `well.to_csv(path)` |

## Curves, units and statistics

Every curve keeps the mnemonic, unit and values found in the file.

```{code-cell} python
import pandas as pd

rows = []
for item in summary["curves"]:
    stats = well[item["mnemonic"]].describe()
    rows.append({"Curve": item["mnemonic"], "Unit": item["unit"],
                 "Valid %": round(100 * item["valid_fraction"]),
                 "Min": stats["min"], "Median": stats["median"], "Max": stats["max"]})
pd.DataFrame(rows).style.hide(axis="index").format(precision=3)
```

Core samples (`PHIC` to `SOC`) fill 1% of depths or less. `SWC` and `SOC` say
`frac` but hold percentages; see {ref}`core-saturation-units`.

## One curve

A curve carries its data, its depths and its metadata.

```{code-cell} python
gr = well["GR"]
print(gr, "| depth unit:", gr.depth_unit, "| source:", gr.metadata.source)
pd.DataFrame([gr.describe()], index=["GR"]).round(2)
```

## Arrays and tables

`well.depth` and `curve.data` are NumPy arrays of the same length, so any
plotting or numerical library can use them.

```{code-cell} python
import matplotlib.pyplot as plt
from petrophys.visualization import use_plot_style
use_plot_style()

brent = well.depth_slice(2841, 2958)          # the Brent reservoir, in metres
fig, axes = plt.subplots(1, 3, figsize=(8, 5), sharey=True)
colors = ["tab:green", "tab:red", "black"]
for ax, name, color in zip(axes, ["GR", "RHOB", "RDEP"], colors):
    ax.plot(brent[name].data, brent.depth, color=color, linewidth=0.8)
    ax.set_xlabel(f"{name} [{brent[name].unit}]")
axes[2].set_xscale("log")
axes[0].set_ylabel(f"Depth [{brent.depth_unit}]")
axes[0].invert_yaxis()
fig.suptitle("30/6-5, Brent reservoir: three curves plotted from their arrays")
plt.show()
```

```{code-cell} python
brent.to_dataframe(curves=["GR", "RHOB", "NPHI", "RDEP"]).head().round(3)
```

## Add a curve

`add_curve` puts a calculated array on the well's depth index. The array must
have one value per depth sample.

```{code-cell} python
import numpy as np

phid = (2.65 - well["RHOB"].data) / (2.65 - 1.0)
well.add_curve("PHID_SS", phid, unit="v/v",
               description="Density porosity, sandstone matrix")
print(well["PHID_SS"], "| curves now:", well.n_curves)

try:
    well.add_curve("TOO_SHORT", np.zeros(10))
except ValueError as error:
    print("ValueError:", error)
```

`pp.interpret` adds its curves the same way, on a copy of the well.
{func}`~petrophys.workflow.run_dynamic_workflow` adds them to the well you pass.

## Depth units and windows

Depths stay in the file's unit. A CSV must declare it, since the file does not
record it. Convert values with {class}`~petrophys.model.UnitConverter`.

```{code-cell} python
from petrophys.model import UnitConverter

print(f"window: {brent.top:.2f} - {brent.bottom:.2f} {brent.depth_unit}, "
      f"{brent.n_samples} samples")
top_ft, bottom_ft = UnitConverter.depth(np.array([brent.top, brent.bottom]), "m", "ft")
print(f"in feet: {top_ft:.1f} - {bottom_ft:.1f} ft")
```

| Input | Depth behaviour |
| --- | --- |
| LAS, CSV, delimited ASCII | Curves stay on the file's depth samples |
| Duplicate or reversing LAS depths | A warning for review; nothing is corrected silently |
| Formation, lithology and core tables | Converted between metres and feet when their unit is declared; a different reference or datum is rejected |
| DLIS with several frames | Compatible channels are linearly resampled onto one grid: grid resampling, not depth matching |
| DLIS image and spectra arrays | Read by the specialist readers, not flattened into the well |
| Independent arrays | Resample explicitly with {func}`~petrophys.model.log_data.align_to_depth` |
| Log-depth matching | Not performed: no tool-offset estimation, pass correlation or stretch-and-squeeze |

## Readers

| Input | Reader | What it builds |
| --- | --- | --- |
| LAS | {func}`~petrophys.io.read_las` | Scalar curves, header, parameters and the file's depth index |
| DLIS (`dlis` extra) | {func}`~petrophys.io.read_dlis` | Compatible scalar channels from the selected frames, on one depth grid |
| CSV, TSV | {func}`~petrophys.io.read_csv` | One well from a wide table sharing a depth column |
| Multiwell CSV | {func}`~petrophys.io.read_csv_wells` | One well per identifier in a long table |
| Delimited ASCII | {func}`~petrophys.io.read_ascii` | A table with a chosen separator, header row, depth column and null value |

`pp.read` picks the reader from the file extension.

## Curve names and units

Vendor mnemonics are recognized and supported units converted. Here four 30/6-5
curves are renamed to Schlumberger-style mnemonics first.

```{code-cell} python
import copy
from petrophys.preprocessing.standardization import standardize_curves

vendor = well.depth_slice(2841, 2958)
schlumberger = {"RHOB": "RHOZ", "NPHI": "TNPH", "RDEP": "AT90", "GR": "GR_EDTC"}
for name, vendor_name in schlumberger.items():
    vendor.rename_curve(name, vendor_name)

standard = standardize_curves(copy.deepcopy(vendor), rename_curves=True,
                              convert_units=True)
changes = pd.DataFrame({
    "In the file": vendor.curves,
    "Unit": [vendor[name].unit for name in vendor.curves],
    "Standard name": standard.curves,
    "Standard unit": [standard[name].unit for name in standard.curves],
})
changed = ((changes["In the file"] != changes["Standard name"])
           | (changes["Unit"] != changes["Standard unit"]))
changes[changed].style.hide(axis="index")
```

The warnings are the core-saturation fix described above. Unknown mnemonics
and units are kept, never guessed. `pp.interpret` keeps vendor names and adds
the standard names beside them; set `rename_input_curves=True` to rename
instead.

## Bundled samples

The package installs small public wells, their geology and the input templates.

```{code-cell} python
from textwrap import fill

names = pp.list_samples()
templates = [name for name in names if name.endswith("_template")]
geology = [name for name in names if name not in templates
           and (name.startswith(("formations_", "lithology_"))
                or name.endswith("_formation_tops"))]
wells = [name for name in names if name not in templates + geology]
for label, group in (("wells", wells), ("geology", geology), ("templates", templates)):
    line = f"{label} ({len(group)}): {', '.join(group)}"
    print(fill(line, 88, subsequent_indent="    "))
```

The Daydream CMR logs are installed as LAS and hold scalar NMR curves only:

```{code-cell} python
cmr = pp.read(pp.sample("daydream1_cmr"))
print(cmr, f"{cmr.top:.0f} - {cmr.bottom:.0f} {cmr.depth_unit}")
print(fill(", ".join(cmr.curves), 88))
```

The large native Daydream DLIS files are not installed. They are Git LFS files
in the source repository; after `git lfs pull`,
{func}`~petrophys.datasets.integration_fixture` returns their paths.

```{code-cell} python
from petrophys.datasets import list_integration_fixtures

list_integration_fixtures()
```

## Go deeper

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Loading and inspecting data
:link: tutorials/part_a/01_loading_and_inspecting_data
:link-type: doc

Headers, curves, coverage and first plots.
:::

:::{grid-item-card} QA/QC and standardization
:link: tutorials/part_a/02_qaqc_and_standardization
:link-type: doc

Aliases, unit fixes, plausibility screens and core units.
:::

:::{grid-item-card} NMR petrophysics
:link: tutorials/part_b/14_nmr_petrophysics
:link-type: doc

The Daydream CMR curves in use.
:::
::::
