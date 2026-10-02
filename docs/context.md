---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Context export

A context is a compact JSON summary of an interpreted well, small enough for an LLM, a report generator or a database.
It holds no depth-by-depth arrays and is built locally, with no LLM or network call. To let an LLM run the interpretation itself, see {doc}`mcp`.

```{code-cell} python
:tags: [hide-cell]
import warnings

import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full

# 30/6-5 stores core saturation in percent under a "frac" label. Petrophys
# converts it and warns; see "Reviewing an interpretation". Only that message is hidden.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## One well

{func}`pp.build_context <petrophys.reporting.context.build_context>` turns an interpretation result into a JSON-safe dictionary:

```{code-cell} python
import json

import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)  # Rw 0.07 ohm.m at 230 degF
context = pp.build_context(result)

def size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

print("total:", size(context), "bytes")
pd.Series({key: size(value) for key, value in context.items()}, name="bytes").to_frame()
```

| Key | Holds |
|---|---|
| `well` | header, depth range, curves with units and valid fraction |
| `interpretation` | status, settings used, key results, net thickness, steps with methods, skip reasons, warnings |
| `geology` | attached formations and lithology, with their import reports |

Each attached formation or group is one entry under `geology`, for example the Brent Group:

```{code-cell} python
formations = context["geology"]["formations"]
brent = next(item for item in formations if item["formation_name"] == "BAAT GP.")
print(json.dumps(brent, indent=1))
```

Pass a result, not a raw well: `build_context` runs no calculations, so a well alone gives no computed curves or trace.

{func}`pp.export_context <petrophys.reporting.context.export_context>` writes the same summary to a file:

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as folder:
    saved = pp.export_context(result, Path(folder, "30_6-5_context.json"))
    print(saved.name, saved.stat().st_size, "bytes")
```

## Many wells: context packages

A package reads a folder of LAS files, interprets each well and writes one JSON per well, formation and lithology group.
Packages interpret as clastic unless told otherwise, so carbonate wells need the carbonate setting. Here are two COSTA wells, labelled with their Kharaib-2 tops:

```{code-cell} python
import shutil

with tempfile.TemporaryDirectory() as folder:
    logs = Path(folder, "logs")
    logs.mkdir()
    for name in ["HW-3", "HW-4"]:
        shutil.copy(pp.sample(name), logs)
    package = Path(folder, "context")
    records = pp.export_context(
        logs, package,
        layout="formation_lithology",
        tops_file=pp.sample("formations_costa"),
        workflow_config=pp.DynamicWorkflowConfig(lithology_domain="carbonate"),
    )
    files = sorted(path for path in package.rglob("*") if path.is_file())
    for path in files:
        name = path.relative_to(package).as_posix()
        print(f"{name:50s} {path.stat().st_size:>7,d} bytes")
    group_file = next(path for path in files if path.parent.name == "KHARAIB-2")
    document = json.loads(group_file.read_text(encoding="utf-8"))
```

Each group file records the interval, the logs used and statistics of every curve. An excerpt from the first one:

```{code-cell} python
from textwrap import fill

analysis = document["analysis_results"]
statistics = analysis["statistics"]
excerpt = {
    "formation_info": document["formation_info"],
    "well_name": analysis["well_name"],
    "lithology group": analysis["lithology group"],
    "input_logs_present": analysis["input_logs_present"],
    "mean": {curve: statistics[f"{curve}_overall"]["arithmetic_mean"]
             for curve in ["VILL", "VCALC", "VDOLO", "PHIE", "SW"]},
    "missing_primary_logs": analysis["data_completeness"]["missing_primary_logs"],
}
for key, value in excerpt.items():
    print(fill(f'"{key}": {json.dumps(value)}', 88, subsequent_indent="    "))
```

| Layout | Files |
|---|---|
| `"well"` (default) | one JSON for one well or result |
| `"formation_lithology"` | `<well>/<formation>/<formation>_<group>.json`, plus the interpreted LAS |
| `"category_formation_lithology"` | `<category>/<well>/<formation>/…_<category>.json` for hydrocarbon, lithology, numerical, permeability, porosity and saturation |

## From the command line

`petrophys-context` always interprets before exporting. Drop any optional flag whose input you do not have.

```bash
petrophys-context well.las well_context.json \
    --formations formation_intervals.csv --lithology lithology_intervals.csv
```

```bash
petrophys-context logs outputs/context_run_01 --layout category_formation_lithology \
    --formations formation_intervals.csv --lithology lithology_intervals.csv \
    --lithology-codes lithology_codes.csv --tops-rank formation
```

| Flag | Use |
|---|---|
| `--well WELL` | export only these wells (repeatable) |
| `--recursive`, `--pattern` | search nested folders; choose which files to read |
| `--min-thickness` | override the grouping threshold |
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
| Formations (`tops_file`) | canonical templates or NPD/SODIR `Well identifier` / `Surface` / `MD`; `tops_options={"sheet_name": ..., "rank": "formation"}` for workbooks |
| Depths | must share the well's reference and datum; metres and feet convert, MD-to-TVD does not |
| Lithology codes (`lithology_mapping_file`) | `code,lithology` or Kingdom `Abbreviation,Attribute`; `Attribute` wins over `Lithology Attribute`, with a warning if they disagree |
| No formations | the logged interval is analyzed as `WHOLE_WELL` |
| No lithology | embedded evidence or the mineral solver supplies it |
| No core curves in LAS | core calibration is skipped and recorded in the trace |
| Thin groups | groups thinner than `minimum_group_thickness` (default 2.0, in each LAS's own depth unit) are not written |
| Carbonate wells | set `lithology_domain` to `carbonate` in LAS metadata, `workflow_config`, or the CLI flag |

## Schema and size contracts

| Contract | Value |
|---|---|
| Single-well schema | `1.0`: `schema_version`, `petrophys_version`, `context_kind`, `well`, `interpretation`, `geology` |
| Missing numbers | JSON `null`, never NaN |
| `max_bytes` | 1,000,000 UTF-8 bytes per JSON document |
| `max_total_bytes` | 50,000,000 bytes per package, ancillary files included |
| Too large | raises {class}`pp.ContextSizeError <petrophys.reporting.context.ContextSizeError>`; nothing is truncated |
| Existing package folder | never overwritten |
| Failed package run | staged and validated first, so no partial folder is left |
| Package documents | written directly, not wrapped in a `payload` object |

## Go deeper

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Generate LLM context
:link: tutorials/part_b/19_llm_context_generation
:link-type: doc
Single-well context and a full category package for 30/6-5.
:::

:::{grid-item-card} Formation and interval analysis
:link: tutorials/part_b/11_formation_and_interval_analysis
:link-type: doc
Nested tops, ranks and per-formation summaries.
:::

:::{grid-item-card} Field screening, 17 COSTA wells
:link: tutorials/case_studies/cs3_field_screening_costa_17wells
:link-type: doc
A carbonate field screened well by well.
:::

:::{grid-item-card} LLM agents (MCP)
:link: mcp
:link-type: doc
Let an LLM run the interpretation itself.
:::
::::
