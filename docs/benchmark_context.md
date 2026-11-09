# Reproducing the benchmark packages

The published context packages of the GeoLink, GeoLink training, COSTA and
Danish collections are computed by dataset context profiles. Each profile
computes the JSON from the input LAS with the collection's interpretation
parameters, calibration and grouping rules. For context from your own wells,
follow the {doc}`context` guide.

## Choose the inputs

| Collection | Profile | Required inputs |
|---|---|---|
| GeoLink | `geolink` (`geolink_v1`) | Raw eval LAS, `NPD stratigraphic picks north sea.xlsx`, and `Litho_Nomenclature_Kingdom.xlsx` |
| GeoLink training | `geolink_train` (`geolink_train_v1`) | Raw `LAS_with_petrophys_params` LAS, `NPD stratigraphic picks north sea.xlsx`, and `Litho_Nomenclature_Kingdom.xlsx` |
| COSTA | `costa` (`costa_v1`) | COSTA LAS; bundled well parameters and reference carbonate volumes are used automatically |
| Danish | `danish` (`danish_v1`) | Raw Danish LAS, `Danish_formation_tops_for_context.xlsx`, and `Litho_Nomenclature_Kingdom.xlsx` |

The profiles cover 32 GeoLink wells, 191 GeoLink training wells, 17 COSTA
wells and five Danish wells. The external GeoLink, GeoLink training and Danish
deliveries/workbooks must be supplied by you.
Inspect supported well names and required inputs in Python:

```python
from petrophys.reporting import describe_context_profile, list_context_profiles

print(list_context_profiles())
print(describe_context_profile("danish"))
```

## What each profile computes

- `geolink_v1` (alias `geolink`) covers the 32 GeoLink eval wells. It reads
  their raw LAS (the `eval` folder), the NPD formation tops workbook and the
  GeoLink lithology codebook, and computes the interpreted curves and category
  context.
- `geolink_train_v1` (alias `geolink_train`) covers the 191 GeoLink training
  wells, from their raw LAS with header interpretation parameters (the
  `LAS_with_petrophys_params` folder), the NPD formation tops workbook and the
  GeoLink lithology codebook.
- `costa_v1` (alias `costa`) covers the 17 COSTA wells. Its inputs ship with
  petrophys: the COSTA LAS files (`petrophys/datasets/costa`), the per-well
  interpretation parameters and reference calcite and dolomite volumes
  (`costa_interpretation_parameters.csv` and
  `costa_reference_carbonate_volumes.csv` in `petrophys/_internal/data`).
  Before the workflow it computes VCL by each well's gamma-ray method, the
  calcite and dolomite volumes and a limestone/dolomite code; the workflow
  takes SW from the source saturation where there is one, and SO = 1 - SW; the
  context notes are the package's per-curve method notes.
- `danish_v1` (alias `danish`) covers five Danish wells, from their raw LAS
  files, the Danish tops workbook and the GeoLink lithology codebook. It first
  writes the prepared LAS the package interpreted (depth in feet, standard
  curve names and unit labels, and a lithology code curve from each
  formation's majority lithology where the well has none), then exports one
  lithology group per formation of the tops workbook, named by its majority
  lithology.

Each well is interpreted once from its raw LAS. A preparatory step, recorded
in the trace, computes the curves the package's settings take from a first
run: for GeoLink and Danish the reservoir flag that selects the SWIRR_APP
samples and the bad-hole flag BADHOLEFLAG, for the GeoLink training package
that reservoir flag, for COSTA its precomputed curves (PERM, VCL, carbonate
volumes and lithology).

A profile fixes its package's settings: it takes `well_names`, `pattern` (LAS
only), `formations` and `lithology_codes`, and rejects `workflow_config` and
the other interpretation options.

## Generate JSON and interpreted LAS from a checkout

Run the following from the repository root in an environment with Petrophys
dependencies installed. For example, install the checkout with
`python -m pip install -e .`. Replace `PATH` with your data location and use a
fresh output directory for each run. The commands use one line so they work
in PowerShell as well as other shells; quote paths containing spaces.

### GeoLink: raw LAS to the full benchmark package

```bash
python scripts/generate_benchmark_context.py --dataset geolink --from-raw --input-dir "PATH/eval" --tops-file "PATH/NPD stratigraphic picks north sea.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/geolink_benchmark_run_01"
```

`--from-raw` is required when a GeoLink **folder** holds raw logs. Without it,
the folder is treated as interpreted LAS, such as `eval_processed`. For a
raw-only run the script first interprets the wells, then exports the context.
The two workbooks default to the parent of the input folder; explicit paths
make the command independent of your directory layout.

### COSTA: bundled raw inputs

```bash
python scripts/generate_benchmark_context.py --dataset costa --input-dir "src/petrophys/datasets/costa" --output-dir "outputs/costa_benchmark_run_01"
```

You can replace the input directory with another copy of the COSTA delivery.
The profile computes VCL, carbonate volumes, interpretation and context with
the collection's settings. It uses bundled reference calcite/dolomite data
where that pipeline requires them.

### Danish: raw LAS and the Danish tops workbook

```bash
python scripts/generate_benchmark_context.py --dataset danish --input-dir "PATH/danish/las" --tops-file "PATH/Danish_formation_tops_for_context.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/danish_benchmark_run_01"
```

The Danish profile sets the input curves and depth units as a preparatory
step, interprets each well, then exports the formation groups used by the
benchmark.

### GeoLink training: raw LAS with header parameters

```bash
python scripts/generate_benchmark_context.py --dataset geolink_train --input-dir "PATH/LAS_with_petrophys_params" --tops-file "PATH/NPD stratigraphic picks north sea.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/geolink_train_run_01"
```

The context's input logs are the logged curves of each LAS: the profile
leaves out the core data the package's LAS carry (`PHIC`, `GDEN`, `KH_CORE`,
`KV_CORE`, `SWC`, `SOC`) and, as the package did, the bit size `BS`, which
also gets no statistics. `processed_las/` holds the interpreted curves at full
precision (17 significant digits), so each statistic computed from them,
mode and geometric and harmonic means included, is the context's statistic.

To write only the interpreted LAS and their workflow traces, without the
context JSON, call the profile's LAS function:

```python
from petrophys.reporting._geolink_train_context import process_geolink_train_las

process_geolink_train_las("PATH/LAS_with_petrophys_params", "outputs/geolink_train_las_01")
```

The statistics of each of those LAS over a context group's samples are the
group's context statistics.

### One well or a subset

`--input-dir` also accepts one LAS file. The script identifies the collection
from the file name, or you can select it explicitly with `--dataset`:

```bash
python scripts/generate_benchmark_context.py --dataset costa --input-dir "src/petrophys/datasets/costa/HW-31.las" --output-dir "outputs/HW-31_benchmark_run_01"
```

For a folder, repeat `--well` to select wells:

```bash
python scripts/generate_benchmark_context.py --dataset costa --input-dir "src/petrophys/datasets/costa" --well HW-31 --well HW-32 --output-dir "outputs/costa_two_wells_run_01"
```

### GeoLink: context from already interpreted LAS

```bash
python scripts/generate_benchmark_context.py --dataset geolink --input-dir "PATH/eval_processed" --raw-input-dir "PATH/eval" --tops-file "PATH/NPD stratigraphic picks north sea.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/geolink_processed_run_01"
```

This reuses the interpreted curves. The raw files identify which curves were
original inputs. Omit `--from-raw` in this mode. Start from the raw inputs when
you also need to regenerate the interpreted LAS.

## What the script writes

For raw profile runs, the output contains six category directories:
`hydrocarbon`, `lithology`, `numerical`, `permeability`, `porosity` and
`saturation`. Each contains
`<well>/<formation>/<formation>_<lithology-group>_<category>.json`.
The root also contains `curve_name_mapping.json`.

The script writes the LAS the context is computed from, with workflow traces,
to `processed_las/`: a context statistic is the statistic of that LAS over the
group's depth window. Each profile interprets each raw LAS once; its traces
record the preparatory step. A `--calculation-profile current` run from raw
LAS keeps its first interpretation in `first_pass_las/`. GeoLink runs that
start with already interpreted LAS reuse those input files rather than
generating this intermediate directory.

Check the printed collection, well count, JSON count and success count after
a run. A nonzero exit status indicates a failure. Use
`python scripts/generate_benchmark_context.py --help` for all options.

## Generate only JSON with the public API

This works after installing Petrophys, without the repository script:

```python
import petrophys as pp

records = pp.export_context(
    None,  # use the bundled COSTA LAS
    "outputs/costa_benchmark_json_01",
    layout="package",
    context_profile="costa",
    well_names=["HW-31", "HW-32"],
)
print(len(records), "context files")
```

For GeoLink, GeoLink training or Danish, replace `None` with the raw LAS directory and provide
`formations` and `lithology_codes` from the input table above.
`pp.export_context` keeps intermediate files in temporary storage and delivers
only the JSON package. Use the repository script to retain processed LAS.

The installed CLI also supports profiles:

```bash
petrophys-context "PATH/danish/las" "outputs/danish_benchmark_json_01" --layout package --profile danish --formations "PATH/Danish_formation_tops_for_context.xlsx" --lithology-codes "PATH/Litho_Nomenclature_Kingdom.xlsx"
petrophys-context "PATH/LAS_with_petrophys_params" "outputs/geolink_train_json_01" --layout package --profile geolink_train --formations "PATH/NPD stratigraphic picks north sea.xlsx" --lithology-codes "PATH/Litho_Nomenclature_Kingdom.xlsx"
```

## Interpret one package well as its context does

`pp.interpret` with `context_profile` runs one well's raw LAS through its
profile's preparation, calculations, value-range QC and dependent-curve
recomputation, and writes nothing. The statistics of `result.well` over a
context group's depths are that group's statistics in the package:

```python
import petrophys as pp

result = pp.interpret(pp.sample("HW-10"), context_profile="costa")
result = pp.interpret("PATH/LAS_with_petrophys_params/15_9-9.las", context_profile="geolink_train")
result = pp.interpret("PATH/eval/15_9-17.las", context_profile="geolink")
result = pp.interpret("PATH/danish/las/ida-1.las", context_profile="danish",
                      formations="PATH/Danish_formation_tops_for_context.xlsx")
```

The well must belong to the profile's package, and the profile fixes the
settings. `result.preparatory_steps` and `result.post_workflow_steps` record
the steps around the workflow; both appear in `result.to_trace_dict()`.

Each context group also gets its category analyses, computed within the
group: indicators, gradients, completeness, electrofacies, and MHI, BVW, RQI,
FZI or PHI_NORM where the interpretation leaves the group without them. They
depend on the group's samples, so they are not curves of `result.well`.
`context_group_statistics` computes a group's full statistics, as its
context files hold them:

```python
from petrophys.reporting.context_profiles import context_group_statistics

info = context_file["analysis_results"]["lithology_group_info"]
stats = context_group_statistics(result, info["start_depth"], info["end_depth"])
stats["saturation"]["MHI_overall"]   # one dict per category file
```

`context_group_record` returns a group's six context documents in full, as
the package writes them: the per-group notes (saturation routing and its
reasons, Archie coefficients per lithology, Rw at temperature, Rcl, the SXO
route, cutoffs per lithology family), curve lists, status and statistics:

```python
from petrophys.reporting.context_profiles import context_group_record

doc = context_file  # one file of the package, for its group's names and depths
info = doc["analysis_results"]
record = context_group_record(
    result, info["lithology_group_info"]["start_depth"], info["lithology_group_info"]["end_depth"],
    formation_name=doc["formation_info"]["name"], group_name=info["lithology group"],
    lithology=info["lithology"], formation_info=doc["formation_info"],
    lithology_mapping_file="PATH/Litho_Nomenclature_Kingdom.xlsx",  # not for COSTA
)
record["saturation"]["analysis_results"]["data_completeness"]["derived_log_computation_notes"]
```

ELECTROFACIES is the exception: the published packages clustered it without a
seed, so only its range is reproducible. The order of the curve names inside a
note may differ from the published files. GeoLink's 15_9-17 and 34_10-23 were
written by a later version of the package's code than its other wells, so
their files also carry SO statistics, two bad-hole fields and some wording
that the profile does not reproduce.

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
matches the intended interval and cutoffs. An MCP server takes the same
configuration with `petrophys-mcp serve --calculation-profile geolink`
({doc}`mcp`).

## Common errors

- A `context_profile` fixes the collection's interpretation and grouping.
  Do not combine it with `workflow_config` or custom interpretation/grouping
  options. Profiles require LAS and one input folder; recursive input is
  unsupported.
- `calculation_profile="geolink"` alone selects equations/defaults. A
  `context_profile` also selects the collection's preparation and export
  pipeline. Use the dataset context profile for benchmark context.
- Unknown or mixed collections cannot be auto-detected. Select the supported
  wells from one collection and supply its required workbooks.
- Use a new output path. The public API rejects existing package directories;
  it supports `overwrite=True` only for single-well JSON files.

Dataset profiles apply the collection's configured interpretation parameters,
calibration and grouping rules.
