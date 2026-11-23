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
agree, then interpret and export. Files cut from the bundled 30/6-5 well stand
in for yours: a CSV of logs, the same tops in feet, and a supplier workbook.

```{code-cell} python
:tags: [remove-cell]
import os, tempfile, textwrap, warnings
import pandas as pd
import petrophys as pp

# Display settings for this page: whole table cells, three decimals, wrapped warnings.
pd.set_option("display.max_colwidth", None)
pd.set_option("display.precision", 3)
warnings.formatwarning = lambda message, category, *args, **kwargs: textwrap.fill(
    f"{category.__name__}: {message}", 88, subsequent_indent="    ") + "\n"
os.chdir(tempfile.mkdtemp())  # the stand-in files and the exports go to a scratch folder
get_ipython().run_line_magic("xmode", "Minimal")  # errors show their message, not the call stack

# Stand-ins for your files, from the bundled 30/6-5 data.
curves = ["GR", "CALI", "BS", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "SP", "PEF", "DRHO"]
logs = pp.read(pp.sample("30_6-5")).depth_slice(2700, 3100).to_dataframe(curves=curves)
logs.rename(columns={"DEPTH": "MD"}).round(4).to_csv("my_logs.csv", index=False)

picks = pd.read_csv(pp.sample("formations_30_6-5")).query("rank == 'formation'")
picks = picks[["formation_name", "top_depth", "bottom_depth"]]
picks.assign(top_depth=picks.top_depth / 0.3048, bottom_depth=picks.bottom_depth / 0.3048
             ).to_csv("formations_ft.csv", index=False)

brent = picks.query("formation_name in ['Ness Fm.', 'Etive Fm.']")
costa = pd.read_csv(pp.sample("formations_costa")).query("well == 'HW-3'")
supplier = pd.DataFrame({
    "Well ID": ["30/6-5"] * len(brent) + ["HW-3"],
    "Marker": [*brent.formation_name, *costa.formation_name],
    "Pick ft": [*(brent.top_depth / 0.3048), *costa.top_depth],
})
with pd.ExcelWriter("supplier_tops.xlsx") as writer:
    supplier.to_excel(writer, sheet_name="Tops", index=False)
    pd.DataFrame({"note": ["Picks from the bundled tops files"]}).to_excel(
        writer, sheet_name="Notes", index=False)
```

## Your logs

The stand-in file, `my_logs.csv`, holds 400 m of 30/6-5 logs: a depth column
called `MD` and eleven log curves, and, like any CSV, no units or header.

```{code-cell} python
import pandas as pd
import petrophys as pp

pd.read_csv("my_logs.csv").head(3)
```

## Read them

A CSV or Excel table does not record its depth unit, so give `depth_unit`, or write the unit in the
depth header (`DEPTH (m)`). The separator and a decimal comma are detected, so a semicolon file
exported in a European locale reads as it is.

```{code-cell} python
well = pp.read("my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5")
well
```

| Option of `pp.read` | Applies to | What it does |
| --- | --- | --- |
| `depth_unit` | CSV, TSV, Excel (required unless the depth header gives the unit) | `"m"` or `"ft"` |
| `depth_column` | CSV, TSV, Excel | Depth column name; default `"DEPTH"`. Headers written `NAME (unit)` or `NAME [unit]`, as {func}`~petrophys.io.export_csv` writes them, give the curve `NAME` in that unit; a depth header's unit must agree with `depth_unit` |
| `well_name` | CSV, TSV, Excel | Name of the well |
| `sheet_name` | Excel | Sheet holding the logs; default the only sheet, else the first with the depth column |
| `null_values`, `skiprows` | CSV, TSV, Excel | Values to read as missing, such as `[-999.25]`, and the number of title rows above the header row |
| `sep`, other `pandas.read_csv` options | CSV, TSV | Passed to {func}`petrophys.io.read_csv` |
| `encoding` | LAS, CSV, TSV | Text encoding of the file, for example `"latin-1"` |
| `frame_name`, `channels`, `step` | DLIS (`dlis` extra) | Frame, channels and grid step; see {func}`petrophys.io.read_dlis` |
| `formations`, `lithology` | all | Geology table: a path or a DataFrame |
| `formation_options`, `lithology_options` | all | `columns`, `rank`, `well_name`, `sheet_name`, `depth_unit`, `depth_reference`, `datum` |
| `depth_reference`, `datum` | all | The well's depth reference and datum, e.g. `"MD"`, `"RKB"` |

Excel logs read the same way: `pp.read("my_logs.xlsx", depth_unit="m")`. A table already
in pandas becomes a well with {meth}`Well.from_dataframe <petrophys.model.Well.from_dataframe>`.
Several wells in one table: {func}`petrophys.io.read_csv_wells`.

## Attach formations and lithology

Pass your tables as `formations=` and `lithology=`. Copy a template's columns
and fill in your picks; {doc}`input_templates` lists every template. Here the
bundled 30/6-5 tops and lithology intervals stand in for yours. `pp.plot` then
draws them beside the logs, as in the {doc}`quickstart`, and the import report
below says what was read.

```{code-cell} python
well = pp.read("my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5",
               formations=pp.sample("formations_30_6-5"),
               lithology=pp.sample("lithology_30_6-5"))
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

### Common service-company curve names

The interpretation workflow recognizes these names and retains the source
curves alongside its canonical working curves by default:

| Measurement | Standard | Examples of recognized names |
| --- | --- | --- |
| Caliper | `CALI` | `HCAL`, `CAL`, `C1`, `C2`, `LCAL`, `CAL1`–`CAL8`, `CALA` (length units required) |
| Density correction | `DRHO` | `HDRA`, `DCOR`, `DRH` |
| Bulk density | `RHOB` | `RHOZ`, `ZDEN`, `RHOM`, `DEN` |
| Neutron porosity | `NPHI` | `TNPH`, `CNL`, `APLC`, `NPHI_LS` |
| Compressional sonic | `DTC` | `DT`, `DTCO`, `AC` |
| Shear sonic | `DTS` | `DTSM`, `DT4S`, `DTRS` |
| Deep resistivity | `RDEP` | `RD`, `ILD`, `IDPH`, `LLD`, `AT90` |
| Medium resistivity | `RMED` | `RM`, `ILM`, `IMPH`, `AT30` |
| Shallow resistivity | `RSHA` | `RS`, `LLS`, `SFLU`, `AT10` |
| Photoelectric factor | `PEF` | `PE`, `PEFZ`, `PEFL` |

Definitions can be checked against the service-company-supplied
[Energistics PWLS catalogue](https://curves.energistics.org/catalog/SLB/curves/)
and the [IODP logging-tool table](https://iodp.tamu.edu/publications/203_IR/chap_02/c2_t4.htm).
Names are not universal: EDE-01 uses `CALA` in inches for ultrasonic caliper,
whereas SLB also defines `CALA` as calibration waveform amplitude. Automatic
caliper selection therefore requires a recognized length unit for `CALA`;
a voltage or missing unit leaves it under its original name. Check the header
before correcting a missing unit. Differential `DCAL` remains subject to the
existing differential-caliper check.

Orthogonal calipers remain separate source curves; selecting a working `CALI`
does not average them. Stoneley `DTST` is not shear `DTS`, and acquisition
suffixes such as EDE-01's `_CH` are not stripped automatically. Verify the
tool, units, neutron matrix reference and acquisition conditions before using
an explicit mapping for a specialized curve.

## Warnings and invalid measurements

Petrophys uses `DataQualityWarning` for suspect numerical inputs and impossible
calculated results. Messages identify the quantity, affected sample count and
up to five zero-based array positions. They say whether values were retained,
replaced, clipped or converted to missing values.

These checks include saturation fractions and resistivity, BVW, NMR permeability,
Timur fallback inputs, core elastic properties, GR baseline ordering, density
contrast and MHI classification. A warning does not repair a retained value:
review the input before using that result. Ordinary missing `NaN` samples alone
do not trigger these numerical range warnings, nor does a porosity of zero,
the effective porosity of shale.

Capture warnings in a user interface or automated job:

```python
import warnings
from petrophys.exceptions import DataQualityWarning

with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter("always", DataQualityWarning)
    result = pp.interpret(well)
messages = [str(item.message) for item in notices]
```

To turn data-quality warnings into exceptions, use
`warnings.simplefilter("error", DataQualityWarning)`. A workflow can record an
exception as a failed step; call `result.raise_for_errors()` when a job must
fail if any step failed. Invalid types, incompatible
survey arrays and unusable scalar parameters raise descriptive `ValueError`
exceptions instead of continuing an undefined calculation.

The core-table reader also records nonnumeric cells converted to `NaN` in
`core.metadata["import_report"]["invalid_numeric_cells"]`, with the source
column, count and zero-based data row positions. Empty cells and recognized
missing-value tokens are distinguished from malformed text.

## Read the import report

Every geology reader returns a table plus a report of what it accepted, mapped
and assumed. Here it notes that no depth reference or datum was declared, and
the rows outside the logged interval.

```{code-cell} python
tops = pp.read_formations(pp.sample("formations_30_6-5"), well=well)
tops.attrs["import_report"]
```

## Check depth reference and datum

Declare the depth unit, reference and datum of each table. Units are then
converted, and a table on a different reference is rejected instead of silently
misplaced. The well declares its own with `depth_reference=` and `datum=`.
Here the same formation picks arrive in feet, in `formations_ft.csv`:

```{code-cell} python
well = pp.read("my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5",
               depth_reference="MD", datum="RKB")
tops = pp.read_formations("formations_ft.csv", well=well,
                          depth_unit="ft", depth_reference="MD", datum="RKB")
tops.attrs["import_report"]["depth_conversion"]
```

```{code-cell} python
tops.tail(3)
```

| Situation | What Petrophys does |
| --- | --- |
| Table in feet, well in metres (or the reverse) | Converts, and records `depth_conversion` in the report |
| No `depth_unit` given | Assumes the well's unit and notes it in the report |
| No `depth_reference` or `datum` given | Assumes they match the well and notes the assumption in the report |
| A `depth_ref` column | Read by its values: a datum (KB, RKB, DF, RT, GL, MSL, SL, LAT) as `datum`, a reference (MD, TVD, TVDSS) as `depth_reference`, `MD RKB` as both |
| A datum the well does not declare | Compared with the datum of the `LMF` (log measured from) item in the LAS `~Well` section; without one there, assumed to match and noted in the report. An MD reference the well does not declare is assumed to match too; TVD or TVDSS raises `ValueError` |
| Different reference (MD, TVD, TVDSS) or datum | Raises `ValueError`: a survey or datum transformation is needed |
| Some rows outside the logged interval | Counts them in the report |
| No row overlaps the logs | Warns with {class}`~petrophys.exceptions.DataQualityWarning`, suggesting the other unit if it fits |

`pp.read` takes the same declarations for its tables through
`formation_options` and `lithology_options`, for example
`formation_options={"depth_unit": "ft", "depth_reference": "MD", "datum": "RKB"}`.

## A supplier workbook

Suppliers often deliver one Excel workbook with several sheets and several
wells, under their own column names. Here the 30/6-5 Brent Group picks share
the `Tops` sheet with the Kharaib-2 pick of the COSTA well HW-3, in feet:

```{code-cell} python
pd.read_excel("supplier_tops.xlsx", sheet_name="Tops")
```

Name the sheet, map the columns (from the template's name to the supplier's)
and pass the well: rows of other wells are filtered out and counted in the
report (`filtered_rows`).

```{code-cell} python
tops = pp.read_formations(
    "supplier_tops.xlsx", sheet_name="Tops", well=well,
    columns={"well": "Well ID", "formation_name": "Marker", "top_depth": "Pick ft"},
    depth_unit="ft", depth_reference="MD", datum="RKB",
)
tops
```

Without `sheet_name`, a workbook with several sheets raises an error that
lists them. Lithology intervals take the same `sheet_name`, `columns` and
`well` options, also from a delimited text file with `sep=";"`; mapped columns
keep vendor comment columns out, and the report lists them.

## Errors you will meet

Petrophys stops with a message that says what to fix. Each call below is a
common mistake, run as it stands.

A CSV without its depth unit:

```{code-cell} python
:tags: [raises-exception]
pp.read("my_logs.csv", depth_column="MD")
```

Tops on another depth reference than the well's:

```{code-cell} python
:tags: [raises-exception]
pp.read_formations("formations_ft.csv", well=well, depth_unit="ft", depth_reference="TVDSS")
```

A misspelt setting:

```{code-cell} python
:tags: [raises-exception]
pp.interpret(well, rw=0.07, rw_temp_f=230)
```

## Interpret

Interpreting, summarizing and exporting work as in the {doc}`quickstart`. What
differs is what your files leave out. A CSV has no header, so there is no
bottom-hole temperature or mud-filtrate resistivity; `result.warnings` and the
skipped steps say what was assumed or left out.

```{code-cell} python
well = pp.read("my_logs.csv", depth_column="MD", depth_unit="m", well_name="30/6-5",
               formations=pp.sample("formations_30_6-5"),
               lithology=pp.sample("lithology_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
result.warnings
```

The second warning is the missing temperature: an assumed 30 °C/km gradient
sets the Rw and Rmf temperature corrections. The `rmf` step did not run:

```{code-cell} python
result.specific_steps_skipped["rmf"]
```

Supply the values with `bht_f` and `rmf_at_meas_temp` (or the mud report's
filtrate chlorides, `rmf_chlorides_mg_l`), or write the CSV well to LAS with
those values in its header ({doc}`input_templates` lists the parameters).

The `input_context` step records where the lithology came from. By default
attached lithology intervals come first, then a recognized lithology log in the
LAS, then the lithology computed from the logs; `lithology_source="external"`,
`"vendor"`, `"computed"` or `"config"` selects one, and a source you request
that is missing fails the run. Here the attached intervals were selected:

```{code-cell} python
result.step("input_context").decisions
```

`result.formation_summary()`, `result.to_las(...)` and `result.to_json(...)`
then give the per-formation answer and the files, as in the {doc}`quickstart`.

## Next

{doc}`capabilities` maps the rest of the library, and {doc}`workflows` takes the interpretation further: the trace, the settings, and the review of an answer before you use it.

The tutorials go deeper on {doc}`loading and inspecting data <tutorials/part_a/01_loading_and_inspecting_data>` and on {doc}`formation and interval analysis <tutorials/part_b/11_formation_and_interval_analysis>`: ranks, nested groups and per-formation results.
