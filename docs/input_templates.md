---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Input templates

Only a well log is required. Six optional tables add formations, lithology,
core data and interpretation settings. Each ships with Petrophys: copy one,
replace its synthetic rows, and read it with the function listed here.

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
import pandas as pd
import petrophys as pp

readers = {
    "formation_tops_template": "pp.read_formations",
    "formation_intervals_template": "pp.read_formations",
    "lithology_intervals_template": "pp.read_lithology_intervals",
    "lithology_codes_template": "pp.read_lithology_codes",
    "well_parameters_template": "pp.read_well_parameters",
    "core_measurements_template": "pp.attach_core_measurements",
}
rows = []
for name in pp.list_samples():
    if name.endswith("_template"):
        columns = list(pd.read_csv(pp.sample(name)).columns)
        more = f" + {len(columns) - 4} more" if len(columns) > 4 else ""
        rows.append({"Template": name, "Read with": readers[name],
                     "Columns": ", ".join(columns[:4]) + more})
pd.DataFrame(rows).style.hide(axis="index")
```

Download:
{download}`formation tops <../src/petrophys/datasets/formation_tops_template.csv>`,
{download}`formation intervals <../src/petrophys/datasets/formation_intervals_template.csv>`,
{download}`lithology intervals <../src/petrophys/datasets/lithology_intervals_template.csv>`,
{download}`lithology codes <../src/petrophys/datasets/lithology_codes_template.csv>`,
{download}`well parameters <../src/petrophys/datasets/well_parameters_template.csv>`,
{download}`core measurements <../src/petrophys/datasets/core_measurements_template.csv>`.
In Python, `pp.sample("<template name>")` gives the installed copy's path.

## What the tables add

Attached to a well, formation and lithology tables become the tops and the
lithology track. The bundled 30/6-5 files follow the same templates.

```{code-cell} python
:tags: [hide-input]
import warnings

# The SWC/SOC unit message is explained in "Core saturation units" below; hide only that.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

```{code-cell} python
well = pp.read(
    pp.sample("30_6-5"),
    formations=pp.sample("formations_30_6-5"),
    formation_options={"rank": "formation"},
    lithology=pp.sample("lithology_30_6-5"),
)
pp.plot(
    well,
    depth_range=(2550, 3150),
    tracks=["LITHOLOGY", "GAMMA / HOLE", "RESISTIVITY", "DENSITY / NEUTRON"],
    figsize=(9.5, 10),
).show()
```

## Rules for every file

| Rule | Detail |
| --- | --- |
| Format | CSV (UTF-8, commas, decimal points), TSV, Excel `.xlsx` or a pandas DataFrame |
| Layout | One header row; no title rows, merged cells or units inside numeric cells |
| Template rows | Synthetic examples, not picks for any well: replace them all |
| Excel workbooks | Several worksheets need `sheet_name` |
| Several wells in one table | Add a `well` column and select with `well_name` or `well` |
| Other column names | Common aliases match case-insensitively; map the rest with `columns={"top_depth": "Pick MD"}` |
| Depth metadata | Optional `depth_unit`, `depth_reference` and `datum` columns or arguments; see {doc}`own_data` |

## Formations

Tops are point picks; intervals add a `bottom_depth` for thickness and summaries.

```{code-cell} python
from IPython.display import display

display(pd.read_csv(pp.sample("formation_tops_template")))
display(pd.read_csv(pp.sample("formation_intervals_template")))
```

The reader maps vendor headers and infers the rank from NPD-style names:

```{code-cell} python
vendor = pd.DataFrame({"Surface": ["Ness Fm.", "Etive Fm."],
                       "MD": [2841.0, 2865.0], "Base MD": [2865.0, 2958.0]})
pp.read_formations(vendor, columns={"formation_name": "Surface", "top_depth": "MD",
                                    "bottom_depth": "Base MD"})
```

A last bottom is never invented. Give `terminal_depth` to turn tops into
intervals:

```{code-cell} python
from petrophys.model import FormationTops

tops = pp.read_formations(pp.sample("formation_tops_template"))
intervals = FormationTops.from_dataframe(tops, terminal_depth=1200.0).to_dataframe()
intervals[["name", "top_depth", "bottom_depth", "thickness"]]
```

| Rule | Behaviour |
| --- | --- |
| Required columns | `formation_name`, `top_depth`; `bottom_depth` is optional |
| Optional columns | `lithology`, `age`, `rank`, `parent`, `well`, `depth_unit`, `depth_reference`, `datum` |
| Header aliases | `formation`, `name`, `surface`, `zone`; `top`, `md`, `depth`, `from`; `bottom`, `base`, `to` |
| Missing name, non-finite depth, `top_depth >= bottom_depth` | Error naming the row |
| Groups and formations mixed | Kept, with their `rank`; select one level with `rank="formation"` |
| Nested levels in summaries | Totals use the union of covered samples, so nesting never double counts |
| Adjacent intervals | `[top, bottom)`; the well's last sample is included |

## Lithology intervals

One rock name per depth interval. Custom names are kept as written.

```{code-cell} python
pd.read_csv(pp.sample("lithology_intervals_template"))
```

Extra vendor columns are rejected unless you map the columns you want; the
report lists what was ignored:

```{code-cell} python
vendor = pd.DataFrame({"FROM": [1000.0, 1025.0], "TO": [1025.0, 1050.0],
                       "ROCK": ["Shale", "Sandstone"], "Comment": ["cuttings", "core"]})
columns = {"top": "FROM", "bottom": "TO", "lithology": "ROCK"}
lithology = pp.read_lithology_intervals(vendor, columns=columns)
print("ignored:", lithology.attrs["import_report"]["ignored_columns"])
lithology
```

| Rule | Behaviour |
| --- | --- |
| Required columns | `top`, `bottom`, `lithology` (aliases `from`, `to`, `base`, `lith`, `rock`, `rocktype`) |
| Rock names | Kept as written; known names get the plot's colours and patterns |
| Overlaps, zero or reversed intervals, empty names | Error naming the row |
| Gaps | Left unfilled |
| Mixtures (several percentages at one depth) | Not accepted; keep them as separate data |
| Precedence | Attached intervals override lithology embedded in the LAS, in plots and `pp.interpret` |

## Lithology codes

A codebook names the codes of a lithology curve already in the LAS. It is not
an interval file.

```{code-cell} python
pd.read_csv(pp.sample("lithology_codes_template"))
```

```{code-cell} python
print(pp.read_lithology_codes(pp.sample("lithology_codes_template")))

kingdom = pd.DataFrame({"Abbreviation": ["SS", "SH", "LS"],
                        "Attribute": ["Sandstone", "Shale", "Limestone"],
                        "Lithology Attribute": ["Clastic", "Clastic", "Carbonate"]})
print(pp.read_lithology_codes(kingdom))
print(pp.read_lithology_codes(kingdom, label_column="Lithology Attribute"))
```

| Rule | Behaviour |
| --- | --- |
| Layouts | `code,lithology`, or Kingdom `Abbreviation,Attribute` |
| `Attribute` and `Lithology Attribute` both present | `Attribute` is used, with a warning if they differ; choose with `label_column` |
| Duplicate codes, empty labels | Error |
| Codes | Vendor-specific; the template's assignments are only examples |
| Use | `pp.plot(well, lithology_mapping=codes)` |

## Well parameters

One row per well records the settings a run used, so it can be repeated.

```{code-cell} python
from textwrap import fill

template = pp.sample("well_parameters_template")
row = pd.read_csv(template).iloc[0]
print(fill(", ".join(f"{name}={value}" for name, value in row.items()), 88))
```

The reader converts friendly names and units into interpretation settings:

```{code-cell} python
settings = pp.read_well_parameters(template, well_name="WELL_001")
print(fill(", ".join(f"{name}={value!r}" for name, value in settings.items()), 88))
# Then: result = pp.interpret(well, **settings)

# The same row in degC and metres gives the same settings:
metric = pd.read_csv(template).assign(bht=104.44, bht_unit="C", td=2590.8, depth_unit="m")
converted = pp.read_well_parameters(metric)
print({name: round(converted[name], 1) for name in ("bht_f", "td_ft")})
```

| Rule | Behaviour |
| --- | --- |
| Column names | Any {doc}`interpretation setting <api/settings>`, plus `gr_clean`, `gr_shale`, `rw_reference_temperature`, `bht`, `td` |
| Units | Temperatures become degF and `td` becomes feet; `rw` must be ohm.m |
| `lithology_domain` | A calculation family (`clastic`, `carbonate`, `auto`), not a per-depth label |
| Blank cell | Uses the LAS header or the workflow default; never another well's value |
| Several rows | Select with `well_name`; duplicates and ambiguous selections are errors |
| Unknown columns, invalid choices | Error |

## Core measurements

One row per plug. The reader converts units and places each plug on the nearest
log sample.

```{code-cell} python
plugs = pd.read_csv(pp.sample("core_measurements_template"))
plugs.T.rename(columns=lambda row: f"plug {row + 1}")
```

The template's plugs are at 8000 ft on a well called `WELL_001`, so a small
matching well receives them here:

```{code-cell} python
import numpy as np

depth = np.arange(7990.0, 8010.0, 0.5)
core_well = pp.Well.from_dataframe(pd.DataFrame({"DEPTH": depth, "GR": 40.0}),
                                   depth_unit="ft", name="WELL_001")
report = pp.attach_core_measurements(core_well, pp.sample("core_measurements_template"))
for key in ["input_rows", "curves", "alignment_tolerance", "depth_unit"]:
    print(f"{key}: {report[key]}")

attached = core_well.to_dataframe(curves=["PHIC", "KH_CORE", "SWC", "SOC", "GDEN"])
attached.dropna(subset=["PHIC"]).style.hide(axis="index").format("{:g}")
```

| Column | Curve | Units accepted |
| --- | --- | --- |
| `core_porosity` | `PHIC` | fraction or percent |
| `core_permeability` | `KH_CORE` (mD) | mD or D |
| `core_water_saturation`, `core_oil_saturation` | `SWC`, `SOC` | fraction or percent |
| `grain_density` | `GDEN` | g/cm3 |
| `depth`, `depth_unit`, `core_to_log_shift` | — | The shift is added before matching. A plug farther than half a sample step from the log is an error. Plugs on the same sample are averaged. |

(core-saturation-units)=
## Core saturation units

`SWC`, `SOC`, `SGC` and their aliases are converted to fractions when a well is
standardized, interpreted or plotted. The bundled 30/6-5 LAS labels `SWC` as a
fraction but stores percentages:

```{code-cell} python
from petrophys.preprocessing.standardization import standardize_core_saturation

swc = pp.read(pp.sample("30_6-5"))["SWC"]
print(f"as stored:  {np.nanmin(swc.data):.3f} - {np.nanmax(swc.data):.3f} {swc.unit}")
standardize_core_saturation(swc, inplace=True)
print(f"normalized: {np.nanmin(swc.data):.3f} - {np.nanmax(swc.data):.3f} {swc.unit}")
record = swc.metadata.custom["core_saturation_normalization"]
{key: value for key, value in record.items() if key != "warnings"}
```

| Case | Result |
| --- | --- |
| Declared percent unit | Divided by 100, even values below 1 |
| No unit, or a fraction label on percent-like values | Percent inferred when at least two samples lie between 1 and 100, at least 80% of positive samples exceed 1, and their median is at least 2; a warning says so |
| A single outlier or a strongly mixed scale | Not rescaled; implausible values are masked with a warning, never clipped |
| Percentages all below 1 with no unit | Cannot be told apart from fractions: confirm, then pass `source_unit="%"` |
| Already normalized | Not divided again |
| Opt out | `standardize_curves(..., convert_units=False)` |

A unit correction does not make core saturation equivalent to log-derived
water saturation.

## All templates at once

The `petrophys-context` command reads every template for one well, or for a
directory of LAS files; see {doc}`context`.

```bash
petrophys-context well.las well_context.json --parameters well_parameters.csv \
    --formations formation_intervals.csv --lithology lithology_intervals.csv \
    --core core_measurements.csv
```

## What to ask your data provider for

| Item | Used for |
| --- | --- |
| Well identifier and the exact LAS used | Matching every table to the well |
| Depth unit, reference (MD, TVD, TVDSS) and datum (RKB, ground level, sea level) | Depth checks |
| Core-to-log depth shift | Placing plugs on the log |
| Tops or top/base intervals, their rank, and the source and version of the picks | Formation summaries |
| Lithology source (core, cuttings or interpretation) and its codebook | The lithology track and matrix choice |
| Core saturation fluid, units and laboratory basis | Core saturation units |

Keep rich vendor exports as the source of record and derive these small tables
from them. Common sources map as follows:

| Source | Layout | How to use it |
| --- | --- | --- |
| [Geocap well tops](https://geocap.atlassian.net/wiki/spaces/ug/pages/22479309/Well%2BTops) | Well, marker name and measured depth | Select the well; map the marker and depth columns |
| [RockWorks lithology](https://help.rockware.com/rockworks17/WebHelp/data_bh_lith.htm) | Top, base and a keyword linked to a dictionary | Lithology intervals plus a codebook |
| [Energistics WITSML lithology](https://energistics.org/sites/default/files/2023-03/WITSML_Lithology_Object_Usage_Guide_1.1Revisions%20Version%202.pdf) | XML intervals with ranks and percentage mixtures | Not read directly; export categorical intervals |
| [NOD factpage for 30/6-5](https://factpages.sodir.no/en/wellbore/PageView/Exploration/By/ByCompletionDate/118) | mMD RKB tops with coincident group and formation picks | Keep the hierarchy in a `rank` column |

## Go deeper

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Your own data
:link: own_data
:link-type: doc

Attach these tables to your logs and check their depths.
:::

:::{grid-item-card} Formation and interval analysis
:link: tutorials/part_b/11_formation_and_interval_analysis
:link-type: doc

Ranks, nested groups and per-formation summaries.
:::

:::{grid-item-card} Core calibration
:link: tutorials/part_b/12_core_calibration_and_digital_core
:link-type: doc

Core porosity and permeability used to calibrate the logs.
:::
::::
