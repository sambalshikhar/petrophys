---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Your own data

The quickstart ran a bundled LAS; your own data rarely arrive that way. Read
your logs, attach formation and lithology tables, check that their depths
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
    top=2800,
    bottom=3000,
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
them. To plot only some curves, pass `pp.plot(well, curves=["GR", "RDEP"])`, by
your names or the standard ones. The visualization masterclass (tutorial 17) and
the visualization guide show both on the Teapot Dome well, whose `SPR` curve no
alias table knows.

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

## A supplier workbook

Suppliers often deliver one Excel workbook with several sheets and several
wells, under their own column names. Name the sheet, map the columns (from the
template's name to the supplier's) and pass the well: rows of other wells are
filtered out and counted in the report. Here the 30/6-5 Brent Group picks share
a sheet with the Kharaib-2 pick of the COSTA well HW-3, in feet:

```{code-cell} python
brent_picks = picks.query("formation_name in ['Ness Fm.', 'Etive Fm.']")
costa_pick = pd.read_csv(pp.sample("formations_costa")).query("well == 'HW-3'")
supplier = pd.DataFrame({
    "Well ID": ["30/6-5"] * len(brent_picks) + ["HW-3"],
    "Marker": [*brent_picks.formation_name, *costa_pick.formation_name],
    "Pick ft": [*(brent_picks.top_depth / 0.3048), *costa_pick.top_depth],
})
with pd.ExcelWriter(workdir / "supplier_tops.xlsx") as writer:
    supplier.to_excel(writer, sheet_name="Tops", index=False)
    pd.DataFrame({"note": ["Picks from the bundled tops files"]}).to_excel(
        writer, sheet_name="Notes", index=False)

tops = pp.read_formations(
    workdir / "supplier_tops.xlsx", sheet_name="Tops", well=well,
    columns={"well": "Well ID", "formation_name": "Marker", "top_depth": "Pick ft"},
    depth_unit="ft", depth_reference="MD", datum="RKB",
)
report = tops.attrs["import_report"]
print(report["filtered_rows"], "row of another well filtered out")
tops.style.hide(axis="index").format(precision=1)
```

Without `sheet_name`, a workbook with several sheets raises an error that
lists them. Lithology intervals take the same `sheet_name`, `columns` and
`well` options, also from a delimited text file with `sep=";"`; mapped columns
keep vendor comment columns out, and the report lists them.

## Errors you will meet

Petrophys stops with a message that says what to fix. Each line below is a real
error raised by the call next to it.

```{code-cell} python
from petrophys.io import read_lithology_intervals


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
show_error(lambda: read_lithology_intervals(rocks.assign(top=rocks.top - 5)))
# A misspelt setting:
show_error(lambda: pp.interpret(well, rw=0.07, rw_temp_f=230))
```

## Interpret, summarize, export

A CSV has no header, so there is no bottom-hole temperature or mud-filtrate
resistivity. The trace and `result.warnings` say what was assumed or skipped.
Supply the values with `bht_f` and `rmf_at_meas_temp` (or the mud report's
filtrate chlorides, `rmf_chlorides_mg_l`), or write the CSV well to LAS with
those values in its header ({doc}`input_templates` lists the parameters).

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

The `input_context` step records where the lithology came from. By default
attached lithology intervals come first, then a recognized lithology log in the
LAS, then the lithology computed from the logs; `lithology_source="external"`,
`"vendor"`, `"computed"` or `"config"` selects one, and a source you request
that is missing fails the run.

```{code-cell} python
source = next(step for step in result.steps_executed
              if step.step == "input_context").parameters
print("lithology source requested / selected:",
      source["requested_source"], "/", source["selected_source"])
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

## Next

{doc}`capabilities` maps the rest of the library, and {doc}`workflows` takes the interpretation further: the trace, the settings, and the review of an answer before you use it.

The tutorials go deeper on {doc}`loading and inspecting data <tutorials/part_a/01_loading_and_inspecting_data>` and on {doc}`formation and interval analysis <tutorials/part_b/11_formation_and_interval_analysis>`: ranks, nested groups and per-formation results.
