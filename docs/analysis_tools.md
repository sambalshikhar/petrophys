---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Analysis tools

Beyond the specialist measurements, Petrophys has functions for analyses
`pp.interpret` does not run: gap filling, clay typing, shale distribution,
matrix and water indicators, heterogeneity, zonation, statistics,
uncertainty, invasion, saturation height and pore pressure. Each
takes arrays or a `Well` and returns plain results; none changes an
interpretation unless you add its output to the well.

Several need data that logs cannot give: core permeability, a free-water
level from formation pressures, true vertical depth from a deviation survey,
the rig and sea-floor elevations, the service company's tool responses and
your input uncertainties. The first section shows how each gets in.

```{code-cell} python
:tags: [remove-cell]

import warnings

warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
# Warnings as one line, without the source file.
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
import pandas as pd

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)
```

The examples use the interpreted 30/6-5 well (Brent Group sands at 2841-2958 m),
its core plugs and its formation tops. The interpreted curves come out as a table,
from which the Brent Group is one query:

```{code-cell} python
import petrophys as pp

result = pp.interpret(pp.sample("30_6-5"), rw=0.070, rw_meas_temp_f=230.0)
logs = result.well.to_dataframe()
brent = logs.query("2841 <= DEPTH <= 2958")
```

## Getting your data in

| You have | Read it with | Used by |
| --- | --- | --- |
| Core analysis table (CSV/Excel) | {func}`~petrophys.io.read_core` | Lorenz, Dykstra-Parsons, Winland, Thomas-Stieber end points |
| Deviation survey (MD, inclination, azimuth) | `pp.read(path, survey=...)` or {func}`~petrophys.io.attach_survey` | Saturation height, pore pressure |
| Rig and datum elevations | the LAS header (EKB, EDF, EGL, APD + EPD by LMF), {func}`~petrophys.io.depth_reference_elevation` | TVDSS, Eaton air gap |
| Formation-tester pressures | {func}`~petrophys.interpretation.saturation.height.read_formation_pressure` | Free-water level |
| Tool-response charts | {func}`~petrophys.interpretation.electrical.invasion.read_tool_responses` | Invasion inversion |
| Input uncertainties (Excel) | {func}`~petrophys.evaluation.read_uncertainty_table` | Monte Carlo |

**Core.** `read_core` finds the columns by name (``Depth (m)``, ``Por (%)``,
``Kh Kl. (mD)``, ``KAIR``, ``Grain density``), converts percent and
percent-labelled-as-fraction porosity, darcy and kg/m3, and converts depths
to the well's unit. Here the bundled 15/12-20 S plug table, which reports
porosity in percent; `CPOR` holds it as a fraction:

```{code-cell} python
from petrophys.io import read_core

core = read_core(pp.sample("epslog_15_12_20_s_core"))
core.to_dataframe().head()
```

The import report names the column found for each property:

```{code-cell} python
core.metadata["import_report"]["columns"]
```

**Pressures and the free-water level.** Station tables are read with their
datum. Labelled stations give the FWL from the water and hydrocarbon
gradients; unlabelled ones, as in the 35/11-13 MDT file, go through
`infer_contacts`, which finds an oil gradient over a water gradient and their
intersection:

```{code-cell} python
from petrophys.interpretation.saturation.height import read_formation_pressure

stations = read_formation_pressure(pp.sample("35_11_13_mdt"))
contacts = stations.infer_contacts()
contacts.status, contacts.fwl_depth, contacts.depth_unit, contacts.depth_reference
```

Every station of the file is used here. Tutorial 15 keeps the 21 tests the file marks
as good and finds the contact at 3113.2 m, against 3112.6 m in the well's report.

**When your file is laid out differently.** Each reader finds columns and
units by name, and every guess can be given instead; a value you give is used
as is:

| Reader | Arguments that override what it reads |
| --- | --- |
| `read_core` | `columns={"depth": "Sample depth", "porosity": "He por", "permeability": "Kair"}`, `depth_unit=`, `porosity_unit="%"`, `permeability_unit="D"`, `sheet_name=` |
| `attach_survey` / `pp.read(..., survey=...)` | `columns={"md": ..., "inclination": ..., "azimuth": ...}`, `elevation=` (the depth reference above sea level, in place of the header's) |
| `read_formation_pressure` | `depth_column=`, `pressure_column=`, `fluid_column=`, `use_column=`, `depth_unit=`, `pressure_unit=`, `depth_reference="TVDSS"`, `fluid_map={1: "gas", 2: "oil", 3: "water"}` for coded fluid columns |
| `read_tool_responses` | `responses={"LLD": "laterolog", "ILD": "induction"}`: the mixing law per tool where the table does not give it (laterolog otherwise) |
| `read_uncertainty_table` | one row per input with `name`, `distribution` (`normal`, `lognormal`, `uniform`, `triangular` or `fixed`) and its parameters (`mean`/`sd`, `median`/`sd`, `low`/`high`, `low`/`mode`/`high`, or `value`), plus optional `low`/`high` bounds for normal and lognormal inputs |

Pretest picking thresholds sit in
{class}`~petrophys.interpretation.saturation.height.MDTPickingConfig`, passed
as `pick_mdt_pressure(..., config=MDTPickingConfig(max_gap_seconds=2.0))`;
each field left unset keeps its default (a time gap over 5 s or five sampling
intervals is flagged for review).

Either result goes straight into
{func}`~petrophys.interpretation.saturation.height.fit_saturation_height_from_well`,
which picks the well's `TVDSS` curve to match the FWL's datum.

**Elevations.** The depth reference's elevation comes from the header,
whichever way the file records it:

```{code-cell} python
from petrophys.io import depth_reference_elevation

{name: depth_reference_elevation(pp.read(pp.sample(name)))
 for name in ("daydream1_cmr", "teapot_44-mdx-10")}
```

**Surveys, tool responses and uncertainties** (no such files are bundled):

```python
well = pp.read("my_well.las", survey="survey.csv")         # adds TVD and, with an elevation, TVDSS
well = pp.read("my_well.las", survey="vertical")           # declare a vertical hole
tools = read_tool_responses("dll_msfl_chart.csv")          # tool, diameter (in/cm), J (0-1 or %), response
inputs = read_uncertainty_table("uncertainties.xlsx")      # name, distribution, min/most likely/max or mean/std
```

A survey table needs MD and inclination (azimuth optional, then constant);
its unit comes from a header such as ``MD (ft)``. Tool responses are the
pseudo-geometric factors J(di) read off the vendor's charts, in a long
table (one row per point) or a wide one (one column per tool).

(analysis-gap-filling)=
## Gap filling with machine learning

{func}`pp.impute <petrophys.preprocessing.imputation.impute>` trains a model on the fly to rebuild a
missing curve from the other logs (it needs the `ml` extra). The result is added as
`<CURVE>_IMPUTED`: the measured values where the curve was logged, the predictions
in its gaps.

On the 223 GeoLink North Sea wells, with 80 % of the wells as `train_wells` and
each curve removed from the 36 others and rebuilt, R² over the rebuilt samples
is 0.79 for RHOB (mean absolute error 0.081 g/cm3), 0.75 for DTC (10.5 us/ft)
and 0.63 for NPHI (0.050 v/v; two wells whose NPHI is zero-filled are not
scored). Scores vary from well to well, so check a rebuild against measured
data in your own wells before using it.

`pp.impute` also writes `<CURVE>_PREDICTED`, a prediction at every depth, so a
rebuild can be checked where the curve was measured. The check is out-of-fold: the
well is cut into five depth blocks, and each measured sample is predicted by a model
that did not learn from its block (the training wells are always used). A model
judged on samples it learned from looks better than it is. Here the density of the
COSTA well HW-24 is predicted from its other logs and the 16 other COSTA wells:

```{code-cell} python
costa = [name for name in pp.list_samples() if name.startswith("HW-") and name != "HW-24"]
hw24 = pp.impute(pp.sample("HW-24"), "RHOB", train_wells=[pp.sample(name) for name in costa])
hw24["RHOB_PREDICTED"].description
```

Over the 433 measured samples the mean absolute error is 0.0346 g/cm3, with almost no
bias (+0.00016 g/cm3) and R² 0.89. The error is given in the curve's unit; for
resistivity and permeability, which span decades, it is given in log10 of the curve.
`pp.plot` draws the prediction over the measured density, in black, in the density track:

```{code-cell} python
pp.plot(hw24, curves=["GR", "RHOB", "RHOB_PREDICTED"])
```

The prediction follows the dense intervals at the top and base and the porous
limestone between them. In a well with gaps, `RHOB_IMPUTED` keeps the measured density
and takes the prediction only in the gaps, where `pp.plot` draws it dashed.

## Matrix and water indicators

Apparent water resistivity, apparent matrix parameters (the inputs of the
MID and RHOmaa-Umaa plots), the Raymer-Hunt-Gardner sonic transform and
Winland R35. Each function takes curves as arrays or table columns.

In clean water-bearing sand, apparent water resistivity (Rwa, Archie solved for Rw
at Sw = 1) reads Rw; clay conduction lowers it in shaly rock. Over the clean
Statfjord water sand it reads 0.089 ohm.m, somewhat above the 0.067 ohm.m that the
interpretation's Rw (0.07 ohm.m at 230 °F) gives at that temperature; in a water leg,
a Rwa well away from Rw questions the Rw or the Archie constants:

```{code-cell} python
from petrophys.interpretation.electrical.rw import apparent_water_resistivity

water = logs.query("3223 <= DEPTH <= 3556 and VCL < 0.15 and PHIE > 0.15")
water.assign(RWA=apparent_water_resistivity(water.RDEP, water.PHIE, a=0.8))[["RWA", "RW"]].median()
```

Apparent matrix density and slowness, from the apparent porosity of density and
neutron together. Quartz reads 2.65 g/cm3 and about 55 us/ft; the Brent Group
medians, 2.74 g/cm3 and 62 us/ft, sit above it, where clay and mica in the sands
move both:

```{code-cell} python
from petrophys.interpretation.lithology import (
    apparent_matrix_density, apparent_matrix_slowness, apparent_total_porosity,
)

phia = apparent_total_porosity(brent.RHOB, brent.NPHI)
brent.assign(RHOMAA=apparent_matrix_density(brent.RHOB, phia),
             DTMAA=apparent_matrix_slowness(brent.DTC, phia))[["RHOMAA", "DTMAA"]].median()
```

Sonic porosity of the Brent Group by the Wyllie time average and by the
Raymer-Hunt-Gardner transform, which does not need a compaction correction:

```{code-cell} python
from petrophys.interpretation.porosity import raymer_hunt_gardner, wyllie_time_average

brent.assign(WYLLIE=wyllie_time_average(brent.DTC),
             RHG=raymer_hunt_gardner(brent.DTC))[["WYLLIE", "RHG"]].median()
```

Winland R35 is the pore-throat radius at 35 % mercury saturation, from core
permeability and porosity; it sorts the Brent plugs into pore-throat classes
(the three plugs without a permeability get no class, the blank row):

```{code-cell} python
import pandas as pd
from petrophys.evaluation.rock_quality import calculate_winland_r35, classify_pore_throat
from petrophys.interpretation.core_analysis import core_data_from_well

plugs = core_data_from_well(result).filter_by_depth(2841, 2958)
r35 = calculate_winland_r35(plugs.permeability, plugs.porosity)
pd.Series(classify_pore_throat(r35)).value_counts()
```

## Shale distribution: Thomas-Stieber

Total porosity against shale volume splits shale into laminae, pore-filling
(dispersed) and grain-replacing (structural) shale. The end points come from
the interval: the clean sand's porosity (90th percentile where VCL is below 0.1)
and the shale's (median where VCL is above 0.8).

```{code-cell} python
from petrophys.interpretation.clay_volume import thomas_stieber

phi_sand = brent.PHIT[brent.VCL < 0.1].quantile(0.9)
phi_shale = brent.PHIT[brent.VCL > 0.8].median()
thomas_stieber(brent.PHIT, brent.VCL, phi_sand, phi_shale)
```

## Clay typing from Th/K

The thorium/potassium ratio of spectral gamma ray separates kaolinite,
mixed-layer clays, illite, micas, glauconite and feldspar (chart bands,
configurable). Here the Daydream 1 HNGS log, a 30 MB file that
`integration_fixture` downloads on first use.

```{code-cell} python
from petrophys.datasets import integration_fixture
from petrophys.interpretation.geochemistry import clay_typing_from_well

clay_typing_from_well(pp.read(integration_fixture("daydream1_shallow_aps_pex_hngs.las")))
```

## Heterogeneity: Lorenz and Dykstra-Parsons

From core permeability and porosity over one reservoir unit, here the Brent plugs.
The Lorenz coefficient runs from 0 (homogeneous) to 1; a Dykstra-Parsons
coefficient above about 0.7 marks a very heterogeneous reservoir.

```{code-cell} python
from petrophys.evaluation import dykstra_parsons, lorenz_coefficient

lorenz_coefficient(plugs.permeability, plugs.porosity), dykstra_parsons(plugs.permeability)
```

## Change-point zonation

PELT finds where the mean of one or more curves shifts. The default penalty
finds bed-scale changes; `max_zones` keeps the strongest boundaries. Compare
the result with the formation tops.

```{code-cell} python
from petrophys.evaluation import zone_well

zones = zone_well(result.well, ["GR", "RHOB"], min_thickness=2.0, top=2700, bottom=3600, max_zones=12)
zones.to_dataframe()[["top", "bottom", "GR_mean", "RHOB_mean"]].round(2)
```

## Statistics by zone

`zone_statistics` tabulates any curves over any zones: a tops file, a table,
or a list of intervals.

```{code-cell} python
from petrophys.evaluation.summaries import zone_statistics

stats = zone_statistics(result.well, pp.sample("formations_30_6-5"), ["PHIE", "SW", "VCL"],
                        rank="formation")
stats.query("zone in ['Ness Fm.', 'Etive Fm.']")[
    ["zone", "curve", "coverage", "arithmetic_mean", "p10", "p50", "p90"]].round(3)
```

## Monte Carlo uncertainty

Each uncertain input gets a distribution; the calculation runs on every
draw. Here mean Archie Sw of the Etive net reservoir with the operator's
constants and assumed spreads: Rw within 20 %, m and n between 1.8 and 2.2, and a porosity
calibration error of 0.015 that is the same at every depth
(`correlated_depths`).

```{code-cell} python
import numpy as np
from petrophys.evaluation import monte_carlo
from petrophys.interpretation.saturation import archie

etive = logs.query("2865 <= DEPTH <= 2958 and NET_RES > 0.5")
rw = etive.RW.median()

def mean_sw(rt, phi, rw, m, n):
    sw = archie(rt=rt, rw=rw, phi=np.clip(phi, 0.01, None), a=0.8, m=m, n=n)
    return np.nanmean(np.clip(sw, 0, 1), axis=-1)

mc = monte_carlo(mean_sw, {
    "rt": etive.RDEP.values, "phi": {"dist": "normal", "mean": etive.PHIE.values, "sd": 0.015},
    "rw": ("triangular", 0.8 * rw, rw, 1.2 * rw), "m": ("uniform", 1.8, 2.2), "n": ("uniform", 1.8, 2.2),
}, n=2000, correlated_depths=["phi"])
mc
```

The rank correlation of each input with the answer shows which one drives the spread:

```{code-cell} python
mc.sensitivity
```

## Invasion, saturation height and pore pressure

These need inputs the bundled data do not state, so the examples are not
run here.

**Step-profile invasion inversion** solves Rt, Rxo and invasion diameter from
three resistivities (or two and a measured Rxo). Its answer depends on each
tool's pseudo-geometric factor J(di): take them from the service company's
charts as {class}`~petrophys.interpretation.electrical.invasion.ToolResponse`
tables. The result carries the range of Rt and di that fits the readings
within their noise, because the step model is often non-unique.

```python
from petrophys.interpretation.electrical.invasion import invert_well_invasion, read_tool_responses

tools = read_tool_responses("dll_chart.csv")           # LLS and LLD from your vendor's charts
inv = invert_well_invasion(well, tools, rxo_curve="MSFL", noise=0.03)
inv                                                    # summary; RT_INV, DI_INV, RT_INV_LOW/HIGH on the well
```

`readings={"LLD": "RDEP", ...}` maps tool names to your curve names.

**Empirical saturation-height functions** fit log Sw against true vertical
height above a known free-water level (power law, Cuddy FOIL, Skelt-Harrison).

```python
from petrophys.interpretation.saturation.height import fit_saturation_height_from_well

well = pp.read("my_well.las", survey="survey.csv")      # TVDSS from the survey and header elevation
fwl = read_formation_pressure("mdt.csv").fit_fwl("oil")  # FWL on TVDSS
result = pp.interpret(well)
shf = fit_saturation_height_from_well(result.well, fwl, model="skelt_harrison", mask=net_reservoir)
shf.parameters, shf.r_squared, shf.predict(height_above_fwl)
```

It warns when the function explains little of Sw (a wrong FWL or datum,
or several rock types fitted together) or its parameters run to their
bounds.

**Eaton pore pressure** needs the overburden (water depth, air gap and the
density above the logged interval), the hydrostatic water density and a
normal compaction trend fitted on clean shale in a normally pressured
section. It is a shale method: give the shale samples as ``shale=`` and the
pressure is computed there, blank elsewhere, and calibration compares each
measured pressure with the shale trend across that depth.

```python
from petrophys.interpretation.pore_pressure import eaton_from_well, fit_normal_compaction_trend

well = pp.read("my_well.las", survey="survey.csv")
shale = well["VCL"].data > 0.6
nct = fit_normal_compaction_trend(tvd, dtc, kind="sonic", mask=shale & normally_pressured, mudline=135)
ppr = eaton_from_well(well, nct, water_depth=110, density_above_log=1.95, shale=shale)  # air gap from the header
exponent, misfit = ppr.calibrate(mdt_tvdss, mdt_bar, datum="TVDSS")      # fit Eaton's exponent
ppr = eaton_from_well(well, nct, water_depth=110, density_above_log=1.95, shale=shale, exponent=exponent,
                      add_curves=True)
```

Offshore the air gap is the depth reference's elevation above sea level;
onshore it is EKB - EGL, and hydrostatic pressure starts at ground level.
Calibrate Eaton's exponent to measured pressures before using the result;
calibration stops with an explanation when the measurements fit no positive
exponent, as with normally pressured points against a poor trend.

## Next

{doc}`context`: hand the interpreted, reviewed well to an LLM as a compact JSON summary.

{doc}`Tutorial 13 <tutorials/part_b/13_ml_curve_imputation>` validates curve imputation in depth, and {doc}`tutorial 15 <tutorials/part_b/15_capillary_pressure_saturation_height>` covers capillary pressure and saturation height.
