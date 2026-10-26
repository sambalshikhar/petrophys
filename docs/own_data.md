---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Your own data

Read your logs, attach formation and lithology tables, check that their depths
agree, then interpret and export. A CSV cut from the bundled 30/6-5 well stands
in for your file.

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

## Your logs

The stand-in file has a depth column called `MD` and eleven log curves, and,
like any CSV, no units or header.

```{code-cell} python
:tags: [hide-input]
import tempfile
from pathlib import Path

import pandas as pd
import petrophys as pp

folder = tempfile.TemporaryDirectory()
workdir = Path(folder.name)

# Stand-in for your file: 400 m of 30/6-5 logs written as a plain CSV.
source = pp.read(pp.sample("30_6-5")).depth_slice(2700, 3100)
curves = ["GR", "CALI", "BS", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "SP", "PEF", "DRHO"]
logs = source.to_dataframe(curves=curves)
logs.rename(columns={"DEPTH": "MD"}).round(4).to_csv(workdir / "my_logs.csv", index=False)

pd.read_csv(workdir / "my_logs.csv").head(3)
```

## Read them

A CSV does not record its depth unit, so `depth_unit` is required.

```{code-cell} python
well = pp.read(workdir / "my_logs.csv", depth_column="MD", depth_unit="m",
               well_name="30/6-5")

print(well)
print(f"{well.top:.2f} - {well.bottom:.2f} {well.depth_unit}, step {well.step:.4f}")
print(well.curves)
```

| Option of `pp.read` | Applies to | What it does |
| --- | --- | --- |
| `depth_unit` | CSV, TSV (required) | `"m"` or `"ft"` |
| `depth_column` | CSV, TSV | Depth column name; default `"DEPTH"` |
| `well_name` | CSV, TSV | Name of the well |
| `null_values`, `skiprows`, `sep`, other `pandas.read_csv` options | CSV, TSV | Passed to {func}`petrophys.io.read_csv` |
| `encoding` | LAS | Text encoding of the file |
| `frame_name`, `channels`, `step` | DLIS (`dlis` extra) | Frame, channels and grid step; see {func}`petrophys.io.read_dlis` |
| `formations`, `lithology` | all | Geology table: a path or a DataFrame |
| `formation_options`, `lithology_options` | all | `columns`, `rank`, `well_name`, `sheet_name`, `depth_unit`, `depth_reference`, `datum` |
| `depth_reference`, `datum` | all | The well's depth reference and datum, e.g. `"MD"`, `"RKB"` |

Excel logs: read them with pandas, then use {meth}`Well.from_dataframe
<petrophys.model.Well.from_dataframe>`. Several wells in one table:
{func}`petrophys.io.read_csv_wells`.

## Attach formations and lithology

Copy a template's columns and fill in your picks; {doc}`input_templates` lists
every template. Here the rows come from the bundled 30/6-5 picks.

```{code-cell} python
formation_columns = pd.read_csv(pp.sample("formation_intervals_template")).columns
picks = pd.read_csv(pp.sample("formations_30_6-5")).query("rank == 'formation'")
picks[formation_columns].to_csv(workdir / "formations.csv", index=False)

lithology_columns = pd.read_csv(pp.sample("lithology_intervals_template")).columns
rocks = pd.read_csv(pp.sample("lithology_30_6-5"))
rocks[lithology_columns].to_csv(workdir / "lithology.csv", index=False)

well = pp.read(
    workdir / "my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5",
    formations=workdir / "formations.csv",
    lithology=workdir / "lithology.csv",
)
viewer = pp.plot(
    well,
    depth_range=(2800, 3000),
    tracks=["LITHOLOGY", "GAMMA / HOLE", "RESISTIVITY", "DENSITY / NEUTRON"],
    figsize=(9.5, 10),
)
viewer.show()
```

## Curve names no alias table knows

Petrophys recognizes the common mnemonics of the main service companies and
software packages, but no table holds every company's names. Map yours once:

```python
pp.read("my_logs.las", curve_aliases={"GAMMA_RAY_1": "GR", "DEEP_RES": "RDEP"})
pp.read("my_logs.las", curve_aliases="my_curve_names.csv")   # columns: curve, standard
```

The mapping is stored on the well, so `pp.interpret` and `pp.plot` use it too;
both also take `curve_aliases=` directly, and `petrophys-context` takes
`--curve-aliases FILE`. Standard names are the library's (`GR`, `RDEP`,
`RHOB`, `NPHI`, `DTC`, `CALI`, ...); an unknown one raises an error that lists
them. To plot only some curves, pass `pp.plot(well, curves=["GR", "RDEP"])`.

## Read the import report

Every geology reader returns a table plus a report of what it accepted, mapped
and assumed. Here it notes the rows outside the logged interval.

```{code-cell} python
from textwrap import fill

tops = pp.read_formations(workdir / "formations.csv", well=well)
report = tops.attrs["import_report"]
print(report["accepted_rows"], "of", report["input_rows"], "rows accepted")
for line in report["diagnostics"]:
    print(fill(f"- {line}", 88, subsequent_indent="  "))
```

## Check depth reference and datum

Declare the depth unit, reference and datum of each table. Units are then
converted, and a table on a different reference is rejected instead of silently
misplaced.

```{code-cell} python
declared = {"depth_unit": "m", "depth_reference": "MD", "datum": "RKB"}
well = pp.read(
    workdir / "my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5",
    depth_reference="MD", datum="RKB",
    formations=workdir / "formations.csv", formation_options=declared,
    lithology=workdir / "lithology.csv", lithology_options=declared,
)

# The same picks, delivered in feet:
in_feet = picks[formation_columns].assign(top_depth=picks.top_depth / 0.3048,
                                          bottom_depth=picks.bottom_depth / 0.3048)
in_feet.to_csv(workdir / "formations_ft.csv", index=False)
tops = pp.read_formations(workdir / "formations_ft.csv", well=well,
                          depth_unit="ft", depth_reference="MD", datum="RKB")
print(tops.attrs["import_report"]["depth_conversion"])
tops.tail(3).style.hide(axis="index").format(precision=1)
```

| Situation | What Petrophys does |
| --- | --- |
| Table in feet, well in metres (or the reverse) | Converts, and records `depth_conversion` in the report |
| No `depth_unit` given | Assumes the well's unit and notes it in the report |
| No `depth_reference` or `datum` given | Assumes they match the well and notes the assumption in the report |
| Different reference (MD, TVD, TVDSS) or datum | Raises `ValueError`: a survey or datum transformation is needed |
| Some rows outside the logged interval | Counts them in the report |
| No row overlaps the logs | Warns with {class}`~petrophys.exceptions.DataQualityWarning`, suggesting the other unit if it fits |

## Errors you will meet

Petrophys stops with a message that says what to fix. Each line below is a real
error raised by the call next to it.

```{code-cell} python
def show_error(action):
    try:
        action()
    except (ValueError, TypeError) as error:
        print(fill(f"{type(error).__name__}: {error}", 88, subsequent_indent="    "))

# No depth unit for a CSV:
show_error(lambda: pp.read(workdir / "my_logs.csv", depth_column="MD"))
# Tops on another depth reference:
show_error(lambda: pp.read_formations(workdir / "formations.csv", well=well,
                                      depth_reference="TVDSS"))
# Overlapping lithology rows:
show_error(lambda: pp.read_lithology_intervals(rocks.assign(top=rocks.top - 5)))
# A misspelt setting:
show_error(lambda: pp.interpret(well, rw=0.07, rw_temp_f=230))
```

## Interpret, summarize, export

A CSV has no header, so there is no bottom-hole temperature or mud-filtrate
resistivity. The trace and `result.warnings` say what was assumed or skipped.
Supply the values with `bht_f` and `rmf_at_meas_temp` (or the mud report's
filtrate chlorides, `rmf_chlorides_mg_l`), or a well-parameters file.

```{code-cell} python
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
print("status:", result.status)

temperature = next(step for step in result.steps_executed if step.step == "temperature")
for label, text in [("temperature", temperature.decisions[0]),
                    ("rmf skipped", result.specific_steps_skipped["rmf"])]:
    print(fill(f"{label}: {text}", 88, subsequent_indent="    "))
for message in result.warnings:
    print(fill(f"warning: {message}", 88, subsequent_indent="    "))
```

Attached formations organize the answer. Averages are over net-pay samples, so
a formation without pay shows no averages.

```{code-cell} python
summary = result.formation_summary()
columns = ["Formation", "Top", "Bottom", "Gross (m)", "Net Pay (m)",
           "Avg PHIE", "Avg Sw", "Avg VCL"]
summary[columns].style.hide(axis="index").format(precision=2, na_rep="–")
```

```{code-cell} python
result.to_las(workdir / "my_well_interpreted.las")       # curves and the settings used
result.to_json(workdir / "my_well_trace.json")           # the complete trace
result.well.to_csv(workdir / "my_well_interpreted.csv")  # curves only

for name in sorted(path.name for path in workdir.iterdir()):
    print(name)
folder.cleanup()
```

## Go deeper

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Input templates
:link: input_templates
:link-type: doc

Every template, its columns and the rules each reader enforces.
:::

:::{grid-item-card} Your own data, in depth
:link: tutorials/own_data
:link-type: doc

Excel workbooks, vendor column names, multiwell tables and bounded intervals.
:::

:::{grid-item-card} Loading and inspecting data
:link: tutorials/part_a/01_loading_and_inspecting_data
:link-type: doc

Curves, headers and quality checks on a freshly read well.
:::

:::{grid-item-card} Formation and interval analysis
:link: tutorials/part_b/11_formation_and_interval_analysis
:link-type: doc

Ranks, nested groups and per-formation results.
:::
::::
