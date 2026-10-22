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

## Choose a workflow

| Task | Entry point |
|---|---|
| Summarize an interpretation you already ran | `pp.build_context(result)` or `pp.export_context(result, "well_context.json")` |
| Interpret your own well and write one JSON | `petrophys-context "well.las" "well_context.json"` |
| Generate six category contexts across wells, formations and lithology groups | `pp.export_context(..., layout="category_formation_lithology")` or the CLI with that layout |
| Reproduce the GeoLink, GeoLink training, COSTA or Danish benchmark | Use the collection's `context_profile`; see {doc}`benchmark_context` for complete commands and required inputs |

For a source checkout, install with `python -m pip install -e .` before using
the public API or installed CLI. For a package installation, use
`python -m pip install petrophys`; DLIS input additionally needs the `dlis`
extra. Run commands from the folder containing your files, or pass absolute
paths. Quote paths containing spaces.

A general category package can be generated with this one-line command:

```bash
petrophys-context "logs" "outputs/context_run_01" --layout category_formation_lithology --formations "formation_intervals.csv" --lithology "lithology_intervals.csv"
```

The two geology flags are optional. Supply your available formation/lithology
data using the formats in {doc}`input_templates`. Use a fresh output directory.
This command interprets the wells with your supplied settings. Select a
dataset profile explicitly when generating benchmark context. The examples below explain configuration, file layouts and checks.

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

A package reads a folder of well files, interprets each well and writes one JSON per well, formation and lithology group. LAS is selected by default; use `pattern="*.dlis"` for DLIS.
Within a formation, a lithology group is a run of consecutive samples with one lithology, taken from the lithology code curve the interpretation read (`LITHOLOGY_GEOLINK` or another lithology-code curve; see {doc}`input_templates`) or from the lithology intervals it selected.
Packages interpret as clastic unless told otherwise, so carbonate wells need the carbonate setting; COSTA wells also take the field's mineral calibration, `carbonate_calibration="costa"`, which writes the illite volume `VILL`. Here are two COSTA wells, labelled with their Kharaib-2 tops:

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
        workflow_config=pp.DynamicWorkflowConfig(lithology_domain="carbonate",
                                                 carbonate_calibration="costa"),
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

Depths are in the well's own unit, feet for HW-3: `formation_info` and `analysis_results.lithology_group_info` name it in `depth_unit`, and the bad-hole intervals in `analysis_results.BADHOLE_QC_CHECK` and the note on KH carry it too.
The statistics are taken after the export's value-range QC, which sets values outside each curve's plausible range to null in every group.
PHI_NORM, RQI, FZI and BVW are then recomputed from the porosity, permeability and saturation that remain, and PHI_Z is set to the recomputed PHI_NORM: each is null wherever the QC removed one of its inputs, so no value computed from a removed sample reaches the statistics.

| Layout | Files |
|---|---|
| `"well"` (default) | one JSON for one well or result |
| `"formation_lithology"` | `<well>/<formation>/<formation>_<group>.json`, plus the interpreted LAS |
| `"category_formation_lithology"` | `<category>/<well>/<formation>/…_<category>.json` for hydrocarbon, lithology, numerical, permeability, porosity and saturation |

## Published packages: GeoLink, COSTA and Danish

For copyable commands covering all three collections, subsets, raw versus
processed LAS, and retained intermediate files, see {doc}`benchmark_context`.

A dataset context profile computes the benchmark package from the raw LAS
using the collection's interpretation parameters, calibration and grouping
rules. Select the profile that matches your dataset.

- `geolink_v1` (alias `geolink`) covers the 32 GeoLink eval wells. It reads
  their raw LAS (the `eval` folder), the NPD formation tops workbook
  (`NPD stratigraphic picks north sea.xlsx`) and the GeoLink lithology
  codebook (`Litho_Nomenclature_Kingdom.xlsx`), and computes the interpreted
  curves and category context.
- `geolink_train_v1` (alias `geolink_train`) covers the 191 GeoLink training
  wells. It reads their raw LAS with header interpretation parameters (the
  `LAS_with_petrophys_params` folder), the NPD formation tops workbook and the
  GeoLink lithology codebook. Each well is interpreted once; a preparatory
  step computes the reservoir flag of the package's first interpretation.
  The input logs are the logged curves, without the core data the LAS carry
  and without the bit size, which the package's context did not list.
- `costa_v1` (alias `costa`) covers the 17 COSTA wells. Its inputs ship with
  petrophys: the COSTA LAS files (`petrophys/datasets/costa`), the per-well
  interpretation parameters and reference calcite and dolomite volumes
  (`costa_interpretation_parameters.csv` and
  `costa_reference_carbonate_volumes.csv` in `petrophys/_internal/data`). Before the workflow it
  computes VCL by each well's gamma-ray method, the calcite and dolomite
  volumes and a limestone/dolomite code; the workflow takes SW from the
  source saturation where there is one, and SO = 1 - SW; the context notes
  are the package's per-curve method notes.
- `danish_v1` (alias `danish`) covers five Danish wells. It reads their raw LAS
  files, the Danish tops workbook (`Danish_formation_tops_for_context.xlsx`)
  and the GeoLink lithology codebook. It first writes the prepared LAS the
  package interpreted (depth in feet, standard curve names and unit labels,
  and a lithology code curve from each formation's majority lithology where
  the well has none), then exports one lithology group per formation of the
  tops workbook, named by its majority lithology.

```python
import petrophys as pp

pp.export_context(
    r"PATH/eval",
    "outputs/geolink_context",
    layout="category_formation_lithology",
    context_profile="geolink",
    tops_file=r"PATH/NPD stratigraphic picks north sea.xlsx",
    lithology_mapping_file=r"PATH/Litho_Nomenclature_Kingdom.xlsx",
)
pp.export_context(
    None,                              # the COSTA LAS bundled with petrophys
    "outputs/costa_context",
    layout="category_formation_lithology",
    context_profile="costa",
)
pp.export_context(
    r"PATH/danish/las",
    "outputs/danish_context",
    layout="category_formation_lithology",
    context_profile="danish",
    tops_file=r"PATH/Danish_formation_tops_for_context.xlsx",
    lithology_mapping_file=r"PATH/Litho_Nomenclature_Kingdom.xlsx",
)
```

```bash
petrophys-context PATH/danish/las outputs/danish_context \
  --layout category_formation_lithology --profile danish \
  --formations PATH/Danish_formation_tops_for_context.xlsx \
  --lithology-codes PATH/Litho_Nomenclature_Kingdom.xlsx
```

A profile fixes its package's settings: it takes `well_names`, `pattern`
(LAS only), `tops_file` and `lithology_mapping_file`, and rejects
`workflow_config` and the other interpretation options. `export_context`
writes the context JSON only. The eval script keeps the LAS too:
`<output-dir>/processed_las` holds the interpreted LAS and workflow traces the
context is computed from, so each context statistic is the statistic of that
LAS over the group's depth window. Each well is interpreted once from its raw
LAS; a preparatory step, recorded in the trace, computes the curves the
package's settings take from a first run: for GeoLink and Danish the reservoir
flag that selects the SWIRR_APP samples and the bad-hole flag BADHOLEFLAG, for
the GeoLink training package that reservoir flag, for COSTA its precomputed
curves (PERM, VCL, carbonate volumes and lithology).
Given one LAS file or a folder, it
recognizes GeoLink, GeoLink training, COSTA or Danish wells by name (`--dataset` overrides):

```bash
python scripts/generate_benchmark_context.py --input-dir PATH/HW-31.las --output-dir outputs/HW-31
```

Use `--well` (or `well_names=[...]`) for a subset, and
`pp.list_context_profiles()` or `pp.describe_context_profile("geolink")` to
inspect the profiles.

## DLIS, CSV and TSV input

Package layouts read each file with `pp.read`, which picks the format from the
extension: LAS, DLIS, CSV or TSV. `pattern` selects the files (default
`"*.las"`), and `reader_options` go to the reader.

DLIS uses the same interpretation and category exporters as LAS. For example,
from a repository checkout, generate all six categories for the deep Daydream-1
delivery:

```python
import petrophys as pp

records = pp.export_context(
    "src/petrophys/datasets/daydream",
    "outputs/daydream_context",
    layout="category_formation_lithology",
    pattern="daydream1_deep_pex_aps_ppc_ecs.dlis",
    reader_options={"step": 0.1524},
)
```

The output directory must be new. `reader_options` are forwarded to `pp.read`;
DLIS supports options such as `channels`, `logical_file`, `channel_preference`
and `step`. The example explicitly merges channels onto a 0.1524 m grid;
omitting `step` uses the reader's default finest grid, which can be much larger.
The step is expressed in the target depth unit (metres by default).
The `formation_lithology` layout also accepts DLIS.

Pass `tops_file` and `lithology_file` just as for LAS to supply real formation
and lithology intervals. Without matched tops, the exporter uses `WHOLE_WELL`;
without lithology information, it records that absence. DLIS alone does not
supply missing geological picks or core measurements.

The corresponding command is:

```bash
petrophys-context src/petrophys/datasets/daydream outputs/daydream_context_cli \
  --layout category_formation_lithology \
  --pattern daydream1_deep_pex_aps_ppc_ecs.dlis --dlis-step 0.1524
```

CSV and TSV files hold one well each, with a depth column (`DEPTH` by
default; `reader_options={"depth_column": "MD"}` names another). They record
no depth unit, so give it: `pattern="*.csv", reader_options={"depth_unit": "m"}`
(CLI: `--pattern "*.csv" --depth-unit m`). They record no curve units or
header parameters either: give the curves in the library's standard units and
pass values such as `rw`, `rw_meas_temp_f` and `bht_f` in `workflow_config`,
which LAS files would carry in their headers. The published-package profiles
read LAS only.

## Class labels

The analysis sections of a group file grade properties into classes, for example clay volume into `CLEAN_FORMATION`, `MODERATE_CLAY` and `SHALY_FORMATION`.
Under the default `"current"` calculations, the classes of clay volume, porosity, permeability, water saturation, BVW, RQI, FZI, MHI, Rxo/Rt and reservoir quality come from one canonical table per property, {py:data}`CLASS_TABLES <petrophys.model.CLASS_TABLES>`, so a property gets the same class in every category file of a group.
A sample without a value gets the table's `NO_*_DATA` label, never a class:

```{code-cell} python
from petrophys.model import CLASS_TABLES

pd.DataFrame([{"table": name, "classes": table.describe()}
              for name, table in CLASS_TABLES.items()]).style.hide(axis="index")
```

Reservoir quality grades effective porosity (PHIE_CC, else PHIE, else PHIT), permeability (PERM_CC, else PERM, PERM_CORE or PERM_TIMUR) and clay volume (VCL, else VCLAY, else VSH) together, with the limits of the lithology family of the group's lithology.

Core-corrected porosity (`PHIE_CC`) needs usable paired log and core porosity
(`PHIC` or `CPOR`), subject to the calibration's sample and fit-quality checks.
It does not require core permeability. Core permeability (`KH_CORE`, `CPERM`
or a supported alias) is needed for the separate `PERM_CC` transform. A well
can therefore export `PHIE_CC` alongside empirical `PERM` without `PERM_CC`.
Limestone, chalk, dolomite and marl are carbonate and every other rock is clastic; a compound name goes by its last rock word, so SANDY LIMESTONE is carbonate and CALCAREOUS SHALE clastic.
A sample without lithology takes the family of the lithology domain of the run that interpreted it.
A sample without effective porosity, permeability or clay volume is `NO_QUALITY_DATA`: it gets no grade, not even `NON_RESERVOIR`.

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
| Formations (`tops_file`) | canonical templates or NPD/SODIR `Well identifier` / `Surface` / `MD`; `tops_options={"sheet_name": ..., "rank": "formation"}` for workbooks |
| Depths | must share the well's reference and datum; metres and feet convert, MD-to-TVD does not |
| Lithology codes (`lithology_mapping_file`) | `code,lithology` or Kingdom `Abbreviation,Attribute`; `Attribute` wins over `Lithology Attribute`, with a warning if they disagree |
| No formations | the logged interval is analyzed as `WHOLE_WELL` |
| No lithology | embedded evidence or the mineral solver supplies it |
| No core curves in LAS | core calibration is skipped and recorded in the trace |
| Thin groups | groups thinner than `minimum_group_thickness` (default 2.0, in each LAS's own depth unit, which each file gives as `formation_info.depth_unit`) are not written |
| Carbonate wells | set `lithology_domain` to `carbonate` in LAS metadata, `workflow_config`, or the CLI flag; for COSTA field wells, also `carbonate_calibration="costa"` in `workflow_config` |

## Configure calculations for an evaluation harness

Use `context_profile="geolink"` for the GeoLink benchmark package. If an
evaluation harness also needs to apply the GeoLink calculation configuration
to other input wells, pass this configuration to `pp.export_context`:

```python
workflow_config=pp.DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True)
```

`reuse_input_resflag=True` retains an input `RESFLAG` or `NET_RES` for the
SWIRR_APP calculation. This is useful when the input LAS already supplies the
reservoir selection used by the evaluation. Check that the supplied flag
matches the intended interval and cutoffs.

## Schema and size contracts

| Contract | Value |
|---|---|
| Single-well schema | `1.0`: `schema_version`, `petrophys_version`, `context_kind`, `well`, `interpretation`, `geology`, and `include_raw_traces_in_context`, always `false` |
| Missing numbers | JSON `null`, never NaN |
| `max_bytes` | 1,000,000 UTF-8 bytes per JSON document |
| `max_total_bytes` | No limit by default; set it to cap a package's total size, ancillary files included |
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
