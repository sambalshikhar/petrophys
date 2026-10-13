---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Input templates

Only a well log is required. Four optional geology tables add formations,
lithology intervals and coded-lithology labels. Each ships with Petrophys:
copy one, replace its synthetic rows, and read it with the function listed here.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import textwrap
import warnings
import numpy as np
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
{download}`lithology codes <../src/petrophys/datasets/lithology_codes_template.csv>`.
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

A codebook names the codes of a lithology code curve already in the LAS. It is
not an interval file.

`pp.interpret` reads one code curve per well: the curve named by the
`lithology_curve` setting, else `LITHOLOGY_GEOLINK`, else the first curve under
a lithology-code name (`LITHOLOGY`, `LITHOFACIES`, `FACIES`, `LITH_CODE`,
`ROCK_TYPE`, ...) whose values pass the code checks: 20 or more finite samples,
whole numbers from 0 to 99 with at most 50 distinct values, and half or more of
the samples GeoLink codes. Other curves whose names contain LITH or FACIES are
not read. Whatever the curve is called, its codes are decoded with the
GeoLink/Kingdom code table, and a warning says so for a curve other than
`LITHOLOGY_GEOLINK`. `pp.interpret` takes no codebook: a codebook names the
codes in plots and context. The "geolink" calculation profile
({ref}`calculation profiles <calculation-profiles>`) keeps the eval package's
search unless `lithology_curve` is set.

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
| Use | `pp.plot(well, lithology_mapping=codes)`; `lithology_mapping_file` in `pp.export_context` |

## LAS metadata and embedded core data

No separate well-parameter or core-measurement template is required. The
workflow reads supported parameters such as Rw, its reference temperature,
BHT, TD, Archie coefficients and zoned values from LAS metadata. Explicit
Python settings take precedence, and documented defaults are used when a value
is absent. The mud and formation-water parameters, and the settings they stand
in for:

| LAS parameter | Setting | How it is read |
| --- | --- | --- |
| `RW`, `RWT` | `rw`, `rw_meas_temp_f` | Rw in ohm.m at RWT; without RWT, Rw is taken at the well's median formation temperature, with a warning |
| `RMF`, `RMFT` | `rmf_at_meas_temp`, `rmf_meas_temp_f` | `MFST`, the mud-filtrate sample temperature, stands in for a missing RMFT |
| `RM`, `RMT` | `rm`, `rm_meas_temp_f` | `MST`, the mud sample temperature, stands in for a missing RMT |
| `CLFL` | `rmf_chlorides_mg_l` | Mud-filtrate chlorides in mg/l (values in ppm, kppm or % are converted). Without an Rmf, Rmf comes from the NaCl equivalent, Cl x 58.44/35.45, divided by the brine density |
| `BHT`, `TD` | `bht_f`, `td_ft` | BHT is taken as the static formation temperature at TD, with no correction for mud circulation |
| `CEC` | `cec` | meq/100 g; a value labelled meq/g is multiplied by 100, and a value under 1 gets a warning, as meq/g values look like that |
| `RWB` | `rwb` | Bound-water resistivity, ohm.m; Dual Water runs only with it |
| `PNSH` with `RBSH` | `phi_cl` (`RBSH` also `rho_cl`) | The neutron reading and the density of shale give the clay porosity; PNSH alone is not used |

Header temperatures labelled degC or K are converted; unlabelled ones are read
as degF. The "geolink" calculation profile ({ref}`calculation profiles
<calculation-profiles>`) reads CLFL as NaCl salinity in ppm, ignores MFST and
MST, and takes PNSH as the clay porosity.

`lithology_domain` defaults to `clastic`. Set it to `carbonate` for carbonate
wells: calcite and dolomite are then solved from bulk density, with clay from
VCL, so the split depends on the pore-fluid density `rho_fl` (1.0 g/cm3 unless
you set it for your brine). For the COSTA field wells,
`carbonate_calibration="costa"` applies the field calibration instead (illite
from GR, a 1.19 g/cm3 brine). Set `auto` when deliberate domain inference is
wanted. A COSTA well with the field calibration:

```{code-cell} python
carbonate = pp.interpret(pp.read(pp.sample("HW-3")), lithology_domain="carbonate",
                         carbonate_calibration="costa")
print(carbonate.config.lithology_domain, carbonate.config.carbonate_calibration)
```

In LAS metadata, use `PP_DOMAIN` or `LITHOLOGY_DOMAIN`. On the command line,
use `--lithology-domain carbonate`. Existing LAS core curves such as `PHIC`,
`KH_CORE`, `SWC`, `SOC` and `GDEN` are standardized in place on the
interpreted copy. Log-derived saturation remains the primary `SW` result;
core saturation is supporting evidence, not an automatic replacement.

The GR equation can also be declared in the LAS parameter section. For example:

```text
GR_METHOD. steiber : Gamma-ray clay-volume transform
GR_MIN.API 20 : Clean-rock gamma-ray endpoint
GR_MAX.API 100 : Shale gamma-ray endpoint
```

`GR_METHOD` accepts `linear`, `larionov_tertiary`, `larionov_older`, `steiber`
and `clavier`; `linear` remains the default. The workflow trace and context
notes record the equation and the endpoints actually applied. These settings
are read per well, including when `--lithology-domain` is supplied. Explicit
Python settings override the LAS values.

The values 20 and 100 API are bounds, not fixed endpoints for every well:
`GRclean = min(GRmin, 20)` and `GRshale = max(GRmax, 100)`. A lower GR minimum
or higher GR maximum is retained; only a minimum above 20 or a maximum below
100 is replaced. For example, selected endpoints of 10.3/71.4 become
10.3/100, while 10/140 stay 10/140. With the module defaults, finite log
minimum/maximum values determine whether the endpoints extend beyond 20/100.
The same bounds apply to explicitly supplied endpoint estimates.

To select the finite GR 5th/95th percentiles instead of the log extrema,
set `GR_MIN` and `GR_MAX` to `auto` in LAS metadata, or pass
`gr_min=None, gr_max=None` to `pp.interpret`. The percentiles are calculated
from the input GR samples on each run, then the same 20/100 bounds apply.

```text
GR_MIN.API auto : Calculate clean endpoint from GR P5
GR_MAX.API auto : Calculate shale endpoint from GR P95
PHIE_CUTOFF.v/v 0.08 : Explicit net-reservoir porosity cutoff
```

`VCL_CUTOFF`, `PHIE_CUTOFF` and `SW_CUTOFF` are optional LAS parameters for
explicit cutoffs. Leaving them absent retains the workflow defaults;
supplying `PHIE_CUTOFF` makes that value apply to every lithology in the well.

When external lithology intervals are selected, context grouping and lithology
summaries use their labels. Gaps stay unclassified even if a different LAS
lithology curve is present. A formation's descriptive majority lithology should
only be supplied as calculation lithology when that interpretation is intended.

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

## Use the optional geology together

The `petrophys-context` command reads a LAS plus either or both optional
geology tables; see {doc}`context`.

```bash
petrophys-context well.las well_context.json \
    --formations formation_intervals.csv --lithology lithology_intervals.csv
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
