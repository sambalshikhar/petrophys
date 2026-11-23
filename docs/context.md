---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
  execution_in_temp: true
---

# Context export

An interpreted and reviewed well can be handed to an LLM. A context is a compact JSON summary of an interpreted well, small enough for an LLM, a report generator or a database.
It holds no depth-by-depth arrays and is built locally, with no LLM or network call. To let an LLM run the interpretation itself, see {doc}`mcp`.

## Choose a workflow

| Task | Entry point |
|---|---|
| Summarize an interpretation you already ran | `pp.export_context(result)` for a dict, or `pp.export_context(result, "well_context.json")` |
| Interpret your own well and write one JSON | `petrophys-context "well.las" "well_context.json"` |
| Generate six category contexts across wells, formations and lithology groups | `pp.export_context(..., layout="package")` or the CLI with that layout |
| Reproduce the GeoLink, GeoLink training, COSTA or Danish benchmark | The collection's `context_profile`; see {doc}`benchmark_context` |

Install Petrophys first ({doc}`installation`); DLIS input also needs the `dlis` extra.
The same exports run from the command line ({ref}`context-cli`). Geology tables are
optional; their layouts are in {doc}`input_templates`.

```{code-cell} python
:tags: [remove-cell]
import warnings

import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full

# 30/6-5 stores core saturation in percent under a "frac" label. Petrophys
# converts it and warns; see "Core saturation units" in the input templates. Only that
# message is hidden.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## One well

Without an output path, {func}`pp.export_context <petrophys.reporting.context.export_context>` turns an interpretation result into a JSON-safe dictionary:

```{code-cell} python
import json
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp="230 degF")  # Rw 0.07 ohm.m at 230 degF
context = pp.export_context(result)
```

Its size in bytes, in all and by top-level key:

```{code-cell} python
len(json.dumps(context)), {key: len(json.dumps(value)) for key, value in context.items()}
```

| Key | Holds |
|---|---|
| `well` | header, depth range, curves with units and valid fraction |
| `interpretation` | status, settings used, key results, net thickness, steps with methods, skip reasons, warnings |
| `geology` | attached formations and lithology, with their import reports |

Each attached formation or group is one entry under `geology`, for example the Brent Group:

```{code-cell} python
next(item for item in context["geology"]["formations"] if item["formation_name"] == "BRENT GP.")
```

Pass a result, not a raw well: a one-well summary runs no calculations, so a well alone gives no computed curves or trace.

Given an output path, `pp.export_context` writes the same summary to a file:

```{code-cell} python
saved = pp.export_context(result, "30_6-5_context.json")
saved.name, saved.stat().st_size
```

## Many wells: context packages

A package reads a folder of well files, interprets each well and writes one JSON per well, formation and lithology group. LAS is selected by default; use `pattern="*.dlis"` for DLIS.
Within a formation, a lithology group is a run of consecutive samples with one lithology, taken from the lithology code curve the interpretation read (`LITHOLOGY_GEOLINK` or another lithology-code curve; see {doc}`input_templates`) or from the lithology intervals it selected.
Packages interpret as clastic unless told otherwise, so carbonate wells need the carbonate setting; COSTA wells also take the field's mineral calibration, `carbonate_calibration="costa"`, which writes the illite volume `VILL`. Here are two of the COSTA wells bundled in one folder, chosen with `well_names` and labelled with their Kharaib-2 tops:

```{code-cell} python
records = pp.export_context(
    pp.sample("HW-3").parent, "context", layout="package", well_names=["HW-3", "HW-4"],
    formations=pp.sample("formations_costa"),
    lithology_domain="carbonate", carbonate_calibration="costa",  # settings as for pp.interpret
)
len(records)
```

The run wrote 168 group files, one record each: six category folders beside a
`curve_name_mapping.json`, and in each the files of every well and formation. The
porosity files of HW-3 hold one lithology group of the Kharaib-2 each:

```{code-cell} python
from pathlib import Path

sorted(path.name for path in Path("context").iterdir())
```

```{code-cell} python
sorted(path.name for path in Path("context/porosity/HW-3/KHARAIB-2").glob("*.json"))
```

Each group file records the interval, the logs used and statistics of every curve. From the first limestone group of HW-3, the interval, the input logs and the statistics of effective porosity:

```{code-cell} python
group = Path("context/porosity/HW-3/KHARAIB-2/KHARAIB-2_LIMESTONE_GROUP_1_porosity.json")
document = json.loads(group.read_text(encoding="utf-8"))
document["formation_info"]
```

```{code-cell} python
document["analysis_results"]["input_logs_present"]
```

```{code-cell} python
document["analysis_results"]["statistics"]["PHIE_overall"]
```

Depths are in the well's own unit, feet for HW-3: `formation_info` and `analysis_results.lithology_group_info` name it in `depth_unit`, and the bad-hole intervals in `analysis_results.BADHOLE_QC_CHECK` and the note on KH carry it too.
The statistics are taken after the export's value-range QC, which sets values outside each curve's plausible range to null in every group; saturations are clipped to 0-1 and permeability to 0.0001-20000 mD, so tight rock stays in the statistics at the 0.0001 mD floor (the benchmark profiles keep their packages' rule, which nulls permeability outside that range).
PHI_NORM, RQI, FZI and BVW are then recomputed from the porosity, permeability and saturation that remain, and PHI_Z is set to the recomputed PHI_NORM: each is null wherever the QC removed one of its inputs, so no value computed from a removed sample reaches the statistics.

| Layout | Files |
|---|---|
| `"well"` (default) | one JSON for one well or result |
| `"package"` | `<category>/<well>/<formation>/…_<category>.json` for hydrocarbon, lithology, numerical, permeability, porosity and saturation |

In a package each formation is split into lithology groups, as above. With `split_by_lithology=False` each formation is one group, `<formation>_<category>.json`, whose documents give its `dominant_lithology`. `zones=` groups the wells by your own intervals in place of the formations, in the same two ways ({doc}`workflows`).

## Published packages

Published packages (GeoLink, GeoLink training, COSTA and Danish) are computed by
their dataset context profiles from the raw LAS, with the collection's
interpretation parameters, calibration and grouping rules; see
{doc}`benchmark_context` for the inputs, commands and checks.

## DLIS, CSV and TSV input

The package layout reads each file with `pp.read`, which picks the format from the
extension: LAS, DLIS, CSV or TSV. `pattern` selects the files (default
`"*.las"`), and `reader_options` go to the reader.

DLIS uses the same interpretation and category exporters as LAS. For example,
generate all six categories for the deep Daydream-1 delivery.
{func}`~petrophys.datasets.integration_fixture` downloads the 173 MB file on
first use and keeps it for later runs:

```python
import petrophys as pp
from petrophys.datasets import integration_fixture

dlis = integration_fixture("daydream1_deep_pex_aps_ppc_ecs.dlis")
records = pp.export_context(
    dlis.parent,
    "outputs/daydream_context",
    layout="package",
    pattern="daydream1_deep_pex_aps_ppc_ecs.dlis",
    reader_options={"step": 0.1524},
)
```

The output directory must be new. `reader_options` are forwarded to `pp.read`;
DLIS supports options such as `channels`, `logical_file`, `channel_preference`
and `step`. The example explicitly merges channels onto a 0.1524 m grid;
omitting `step` uses the reader's default finest grid, which can be much larger.
The step is expressed in the target depth unit (metres by default).

Pass `formations` and `lithology` just as for LAS to supply real formation
and lithology intervals. Without matched tops, the exporter uses `WHOLE_WELL`;
without lithology information, it records that absence. DLIS alone does not
supply missing geological picks or core measurements.

The corresponding command, given the folder that holds the file
(`integration_fixture(...).parent`), is:

```bash
petrophys-context <folder> outputs/daydream_context_cli \
  --layout package \
  --pattern daydream1_deep_pex_aps_ppc_ecs.dlis --dlis-step 0.1524
```

CSV and TSV files hold one well each, with a depth column (`DEPTH` by
default; `reader_options={"depth_column": "MD"}` names another). They record
no depth unit, so give it: `pattern="*.csv", reader_options={"depth_unit": "m"}`
(CLI: `--pattern "*.csv" --depth-unit m`). They record no curve units or
header parameters either: give the curves in the library's standard units and
pass values such as `rw`, `rw_meas_temp` and `bht` as settings, which LAS
files would carry in their headers. The published-package profiles
read LAS only.

## Class labels

The analysis sections of a group file grade properties into classes, for example clay volume into `CLEAN_FORMATION`, `MODERATE_CLAY` and `SHALY_FORMATION`.
Under the default `"current"` calculations, the classes of clay volume, porosity, permeability, water saturation, BVW, RQI, FZI, MHI, Rxo/Rt and reservoir quality come from one canonical table per property, {py:data}`CLASS_TABLES <petrophys.model.CLASS_TABLES>`, so a property gets the same class in every category file of a group.
A sample without a value gets the table's `NO_*_DATA` label, never a class:

```{code-cell} python
from petrophys.model import CLASS_TABLES

pd.Series({name: table.describe() for name, table in CLASS_TABLES.items()}).to_frame("classes")
```

Reservoir quality grades effective porosity (PHIE_CC, else PHIE, else PHIT), permeability (PERM_CC, else PERM, PERM_CORE or PERM_TIMUR) and clay volume (VCL, else VCLAY, else VSH) together, with the limits of the lithology family of the group's lithology.

Core-corrected porosity (`PHIE_CC`) needs usable paired log and core porosity
(`PHIC` or `CPOR`), subject to the calibration's sample and fit-quality checks.
It does not require core permeability. Core permeability (`KH_CORE`, `CPERM`
or a supported alias) is needed for the separate `PERM_CC` transform. A well
can therefore export `PHIE_CC` alongside empirical `PERM` without `PERM_CC`.
Limestone, chalk, dolomite and marl are carbonate and every other rock takes the clastic family; a compound name goes by its last rock word, so SANDY LIMESTONE is carbonate and CALCAREOUS SHALE clastic.
A sample without lithology takes the family of the lithology domain of the run that interpreted it.
A sample without effective porosity, permeability or clay volume is `NO_QUALITY_DATA`: it gets no grade, not even `NON_RESERVOIR`.

(context-cli)=
## From the command line

`petrophys-context` always interprets before exporting. Drop any optional flag whose input you do not have.

```bash
petrophys-context well.las well_context.json \
    --formations formation_intervals.csv --lithology lithology_intervals.csv
```

```bash
petrophys-context logs outputs/context_run_01 --layout package \
    --formations formation_intervals.csv --lithology lithology_intervals.csv \
    --lithology-codes lithology_codes.csv --tops-rank formation
```

| Flag | Use |
|---|---|
| `--well WELL` | export only these wells (repeatable) |
| `--recursive`, `--pattern` | search nested folders; choose which files to read |
| `--min-thickness` | override the grouping threshold |
| `--setting NAME=VALUE` | an interpretation setting of `petrophys.interpret`, e.g. `--setting rw=0.07 --setting rw_meas_temp="230 degF"` (repeatable; the value is read as JSON, else as text); not with `--profile` |
| `--profile` | compute the published `geolink`, `geolink_train`, `costa` or `danish` package from its raw LAS (geolink, geolink_train and danish also need `--formations` and `--lithology-codes`); omit it to interpret with your supplied settings |
| `--tops-rank`, `--tops-sheet` | pick the formation level (`group`, `formation`, `member`) and the Excel sheet |
| `--lithology-domain` | broad mineral-solving domain: `clastic`, `carbonate`, `evaporite` or `auto` |
| `--max-bytes`, `--max-total-bytes` | change the size limits |
| `--depth-unit` | required for CSV/TSV logs |
| `--overwrite` | replace an existing single-well JSON (packages are never overwritten) |

`python -m petrophys.reporting.context_cli` is equivalent.

The first command in Python, with your own files:

```python
well = pp.read("well.las", formations="formation_intervals.csv",
               lithology="lithology_intervals.csv")
result = pp.interpret(well)  # reads supported settings and core curves from LAS
pp.export_context(result.raise_for_errors(), "well_context.json")
```

## Multi-well input rules

The standardized readers accept CSV, TSV, Excel and DataFrames; {doc}`input_templates` lists the columns.

| Input | Rule |
|---|---|
| Matching | each table row matches a LAS file by the well name in its header, ignoring separators (`30/6-5` matches `30_6-5`) |
| Tables covering several wells | need a `well` column; without one, the folder may hold only one LAS |
| Formations (`formations`) | canonical templates or NPD/SODIR `Well identifier` / `Surface` / `MD`; `tops_options={"sheet_name": ..., "rank": "formation"}` for workbooks |
| Depths | must share the well's reference and datum; metres and feet convert, MD-to-TVD does not |
| Lithology codes (`lithology_codes`) | `code,lithology` or Kingdom `Abbreviation,Attribute`; `Attribute` wins over `Lithology Attribute`, with a warning if they disagree |
| No formations | the logged interval is analyzed as `WHOLE_WELL` |
| No lithology | embedded evidence or the mineral solver supplies it |
| No core curves in LAS | core calibration is skipped and recorded in the trace |
| Thin groups | groups thinner than `minimum_group_thickness` (default 2.0, in each LAS's own depth unit, which each file gives as `formation_info.depth_unit`) are not written |
| Carbonate wells | set `lithology_domain` to `carbonate` in LAS metadata, as a setting, or with the CLI flag; for COSTA field wells, also the setting `carbonate_calibration="costa"` |

## Schema and size contracts

| Contract | Value |
|---|---|
| Single-well schema | `1.0`: `schema_version`, `petrophys_version`, `context_kind`, `well`, `interpretation`, `geology`, and `include_raw_traces_in_context`, always `false` |
| Missing numbers | JSON `null`, never NaN |
| `max_bytes` | 1,000,000 UTF-8 bytes per JSON document |
| `max_total_bytes` | No limit by default; set it to cap a package's total size, ancillary files included |
| Too large | raises {class}`ContextSizeError <petrophys.reporting.context.ContextSizeError>` (from `petrophys.reporting`); nothing is truncated |
| Existing package folder | never overwritten |
| Failed package run | staged and validated first, so no partial folder is left |
| Package documents | written directly, not wrapped in a `payload` object |

## Next

{doc}`mcp`: let an LLM run the interpretation itself, instead of reading its result.

{doc}`Tutorial 19 <tutorials/part_b/19_llm_context_generation>` builds a single-well context and a full category package for 30/6-5.
