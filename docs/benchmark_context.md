# Generate benchmark context

Use a dataset profile to generate benchmark context for GeoLink, COSTA or
Danish wells. Each profile computes JSON from the input LAS with the dataset's
interpretation parameters, calibration and grouping rules. For context from
your own wells, follow the {doc}`context` guide.

## Choose the inputs

| Collection | Profile | Required inputs |
|---|---|---|
| GeoLink | `geolink` (`geolink_v1`) | Raw eval LAS, `NPD stratigraphic picks north sea.xlsx`, and `Litho_Nomenclature_Kingdom.xlsx` |
| COSTA | `costa` (`costa_v1`) | COSTA LAS; bundled well parameters and reference carbonate volumes are used automatically |
| Danish | `danish` (`danish_v1`) | Raw Danish LAS, `Danish_formation_tops_for_context.xlsx`, and `Litho_Nomenclature_Kingdom.xlsx` |

The profiles cover 32 GeoLink wells, 17 COSTA wells and five Danish wells.
The external GeoLink and Danish deliveries/workbooks must be supplied by you.
Inspect supported well names and required inputs in Python:

```python
import petrophys as pp

print(pp.list_context_profiles())
print(pp.describe_context_profile("danish"))
```

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
original inputs. Omit `--from-raw` in this mode. Using raw inputs for a complete
reproduction is preferable when you also need to regenerate interpreted LAS.

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
    layout="category_formation_lithology",
    context_profile="costa",
    well_names=["HW-31", "HW-32"],
)
print(len(records), "context files")
```

For GeoLink or Danish, replace `None` with the raw LAS directory and provide
`tops_file` and `lithology_mapping_file` from the input table above.
`pp.export_context` keeps intermediate files in temporary storage and delivers
only the JSON package. Use the repository script to retain processed LAS.

The installed CLI also supports profiles:

```bash
petrophys-context "PATH/danish/las" "outputs/danish_benchmark_json_01" --layout category_formation_lithology --profile danish --formations "PATH/Danish_formation_tops_for_context.xlsx" --lithology-codes "PATH/Litho_Nomenclature_Kingdom.xlsx"
```

## Reproduction limits and common errors

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
- Reference ELECTROFACIES clustering was unseeded, so those statistics may
  differ. Reference comparisons also record SO/note differences for GeoLink's
  15/9-17 and 34/10-23, and three SW sample differences for 16/10-4. These are
  reproduction limits; byte-identical output is not guaranteed.

Dataset profiles apply the collection's configured interpretation parameters,
calibration and grouping rules.
