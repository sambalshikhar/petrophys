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
geology. This page takes the bundled 30/6-5 well apart: 22,176 depth samples
0.152 m apart, from 176.6 m to 3556.1 m. The header values the interpretation
reads are the bottom-hole temperature (degC), the formation-water resistivity and
its temperature (degF), and the mud-filtrate resistivity:

```{code-cell} python
:tags: [remove-cell]
import textwrap
import warnings
import pandas as pd

# Display settings for this page: whole table cells, three decimals, wrapped warnings.
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.3f}".format)
warnings.formatwarning = lambda message, category, *args, **kwargs: textwrap.fill(
    f"{category.__name__}: {message}", 88, subsequent_indent="    ") + "\n"
# 30/6-5 stores core saturation in percent under a "frac" label; Petrophys converts it
# and warns (see "Core saturation units" in the input templates). Only that message is hidden.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

```{code-cell} python
import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))
well.describe()["header_parameters"]
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
`well.describe()` lists them, with the fraction of depths each one covers:

```{code-cell} python
pd.DataFrame(well.describe()["curves"])
```

Core samples (`PHIC` to `SOC`) fill 1% of depths or less. `SWC` and `SOC` say
`frac` but hold percentages; see {ref}`core-saturation-units`.

## One curve

A curve carries its data, its depths and its metadata.

```{code-cell} python
well["GR"]
```

```{code-cell} python
well["GR"].describe()
```

The raw gamma ray reaches -12.6 API, a value no formation gives; `pp.interpret`
screens implausible values before it uses a curve ({ref}`review-inputs`).

## Arrays and tables

`well.depth` and `curve.data` are NumPy arrays of the same length, so any
plotting or numerical library can use them. `depth_slice` cuts a depth window
into a new well, and `to_dataframe` gives a table with one row per depth:

```{code-cell} python
brent = well.depth_slice(2841, 2958)          # the Brent Group, in metres
brent.to_dataframe(curves=["GR", "RHOB", "NPHI", "RDEP"]).head()
```

## Add a curve

`add_curve` puts an array on the well's depth index, for example a curve from
another source on the same depths. Here the effective porosity of an
interpretation goes back on the raw well:

```{code-cell} python
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
well.add_curve("PHIE", result.well["PHIE"].data, unit="v/v", description="Effective porosity")
well["PHIE"]
```

The array must have one value per depth sample: an array of any other length is
refused with a `ValueError` that gives both lengths.

```{code-cell} python
try:
    well.add_curve("TOO_SHORT", result.well["PHIE"].data[:10])
except ValueError as error:
    print(error)
```

`pp.interpret` adds its curves the same way, on a copy of the well.
{func}`~petrophys.workflow.run_dynamic_workflow` adds them to the well you pass.

## Depth units and windows

Depths stay in the file's unit. A CSV must declare it, since the file does not
record it. Convert values with {class}`~petrophys.model.UnitConverter`.

```{code-cell} python
from petrophys.model import UnitConverter

brent.top, brent.bottom, brent.depth_unit
```

```{code-cell} python
UnitConverter.depth(brent.depth[[0, -1]], "m", "ft")
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

Vendor mnemonics are recognized and supported units converted. The Teapot Dome
well 17-WX-21 is logged under its contractor's mnemonics (`GRD`, `CALD`, `ILD`,
...) and units (`G/C3`, `DEC`, `US/F`):

```{code-cell} python
from petrophys.preprocessing.standardization import standardize_curves

teapot = pp.read(pp.sample("teapot_17-wx-21"))
standard = standardize_curves(pp.read(pp.sample("teapot_17-wx-21")), rename_curves=True)
pd.DataFrame({"In the file": teapot.curves, "Unit": [teapot[name].unit for name in teapot.curves],
              "Standard": standard.curves, "Standard unit": [standard[name].unit for name in standard.curves]})
```

Unknown mnemonics and units are kept, never guessed. `pp.interpret` keeps
vendor names and adds the standard names beside them; set
`rename_input_curves=True` to rename instead.

## Bundled samples

The package installs small public datasets: well logs, NMR, formation-pressure
tests and core measurements, with geology tables and input templates.
`pp.list_samples(kind)` lists one kind, or one collection such as `"costa"`:

```{code-cell} python
from petrophys.datasets import SAMPLE_KINDS

pd.Series({kind: pp.list_samples(kind) for kind in SAMPLE_KINDS})
```

The Daydream CMR logs are installed as LAS and hold scalar NMR curves only,
from 3755 m to 4145 m:

```{code-cell} python
pp.read(pp.sample("daydream1_cmr")).curves
```

The large native Daydream DLIS files are not installed:
{func}`~petrophys.datasets.integration_fixture` downloads one on first use,
checks its SHA-256, keeps it in `petrophys.datasets.data_dir()` (set
`PETROPHYS_DATA_DIR` to choose the folder) and returns its path. A source
checkout with `git lfs pull` uses its own copies instead. Reading DLIS needs
`pip install "petrophys[dlis]"`; the inventory and checksums are in
`src/petrophys/datasets/daydream/README.md`.

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
