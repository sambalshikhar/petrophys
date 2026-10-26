---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Analysis tools

Functions for analyses `pp.interpret` does not run: clay typing, shale
distribution, matrix and water indicators, heterogeneity, zonation,
statistics, uncertainty, invasion, saturation height and pore pressure. Each
takes arrays or a `Well` and returns plain results; none changes an
interpretation unless you add its output to the well.

Several need data that logs cannot give: core permeability, a free-water
level from formation pressures, true vertical depth from a deviation survey,
the rig and sea-floor elevations, the service company's tool responses and
your input uncertainties. The first section shows how each gets in.

```{code-cell} python
:tags: [remove-cell]

%matplotlib inline
import warnings

warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
import pandas as pd

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)
```

The examples use the interpreted 30/6-5 well (Brent sands at 2841-2958 m),
its core plugs and its formation tops.

```{code-cell} python
import numpy as np
import petrophys as pp

result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.070, rw_meas_temp_f=230.0)
well = result.well
depth = np.asarray(well.depth)
brent = (depth >= 2841) & (depth <= 2958)
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
porosity in percent:

```{code-cell} python
from petrophys.io import read_core

core = read_core(pp.sample("epslog_15_12_20_s_core"))
print(core, "| columns:", core.metadata["import_report"]["columns"])
print(f"porosity median {np.nanmedian(core.porosity):.3f} v/v")
```

**Pressures and the free-water level.** Station tables are read with their
datum. Labelled stations give the FWL from the water and hydrocarbon
gradients; unlabelled ones, as in the 35/11-13 MDT file, go through
`infer_contacts`:

```{code-cell} python
from petrophys.interpretation.saturation.height import read_formation_pressure

stations = read_formation_pressure(pp.sample("35_11_13_mdt"))
contacts = stations.infer_contacts()
print(contacts.status, [(c.kind, round(c.depth, 1)) for c in contacts.contacts],
      stations.depth_unit, stations.depth_reference)
```

Either result goes straight into
{func}`~petrophys.interpretation.saturation.height.fit_saturation_height_from_well`,
which picks the well's `TVDSS` curve to match the FWL's datum.

**Elevations.** The depth reference's elevation comes from the header,
whichever way the file records it:

```{code-cell} python
from petrophys.io import depth_reference_elevation

for name in ("daydream1_cmr", "teapot_44-mdx-10"):
    print(name, depth_reference_elevation(pp.read(pp.sample(name))))
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

## Matrix and water indicators

Apparent water resistivity, apparent matrix parameters (the inputs of the
MID and RHOmaa-Umaa plots), the Raymer-Hunt-Gardner sonic transform and
Winland R35.

```{code-cell} python
from petrophys.evaluation.rock_quality import calculate_winland_r35, classify_pore_throat
from petrophys.interpretation.electrical.rw import apparent_water_resistivity
from petrophys.interpretation.lithology import (
    apparent_matrix_density, apparent_matrix_slowness, apparent_total_porosity,
)
from petrophys.interpretation.porosity import raymer_hunt_gardner, wyllie_time_average

rhob, nphi, dtc = (np.asarray(well.get(n).data)[brent] for n in ("RHOB", "NPHI", "DTC"))
phia = apparent_total_porosity(rhob, nphi)
# Rwa in the clean Statfjord water sand reads Rw (clay conduction lowers it in shaly rock).
curve = lambda name: np.asarray(well.get(name).data)
water = (depth >= 3223) & (depth <= 3556) & (curve("VCL") < 0.15) & (curve("PHIE") > 0.15)
rwa = apparent_water_resistivity(curve("RDEP")[water], curve("PHIE")[water], a=0.8)
print(f"Statfjord clean sand: Rwa median {np.nanmedian(rwa):.3f} ohm.m; "
      f"Rw of the interpretation {np.nanmedian(curve('RW')[water]):.3f} ohm.m")
print(f"RHOmaa median {np.nanmedian(apparent_matrix_density(rhob, phia)):.2f} g/cc, "
      f"DTmaa median {np.nanmedian(apparent_matrix_slowness(dtc, phia)):.1f} us/ft")
print(f"Sonic porosity at 80 us/ft: Wyllie {wyllie_time_average(80.0):.3f}, RHG {raymer_hunt_gardner(80.0):.3f}")

core = np.isfinite(np.asarray(well.get("KH_CORE").data)) & brent
r35 = calculate_winland_r35(np.asarray(well.get("KH_CORE").data)[core], np.asarray(well.get("PHIC").data)[core])
print(pd.Series(classify_pore_throat(r35)).value_counts().to_string())
```

## Shale distribution: Thomas-Stieber

Total porosity against shale volume splits shale into laminae, pore-filling
(dispersed) and grain-replacing (structural) shale. The end points come from
the interval: the clean sand's porosity and the shale's.

```{code-cell} python
from petrophys.interpretation.clay_volume import thomas_stieber

phit, vcl = (np.asarray(well.get(n).data)[brent] for n in ("PHIT", "VCL"))
phi_sand = float(np.nanpercentile(phit[vcl < 0.1], 90))
phi_shale = float(np.nanmedian(phit[vcl > 0.8]))
ts = thomas_stieber(phit, vcl, phi_sand, phi_shale)
print(f"phi_sand {phi_sand:.3f}, phi_shale {phi_shale:.3f}; sand-bed fraction (N/G) "
      f"{np.nanmean(ts.net_to_gross):.2f}; samples outside the model {int(np.sum(ts.regime == 0))}")
```

## Clay typing from Th/K

The thorium/potassium ratio of spectral gamma ray separates kaolinite,
mixed-layer clays, illite, micas, glauconite and feldspar (chart bands,
configurable). Here the Daydream 1 HNGS log.

```{code-cell} python
from petrophys.datasets import integration_fixture
from petrophys.interpretation.geochemistry import clay_typing_from_well

daydream = pp.read(integration_fixture("daydream1_shallow_aps_pex_hngs.las"))
typing = clay_typing_from_well(daydream)
print("From", typing.sources, f"- {typing.classified} samples typed")
print({name: round(share, 3) for name, share in typing.fractions.items() if share})
```

## Heterogeneity: Lorenz and Dykstra-Parsons

From core permeability and porosity over one reservoir unit.

```{code-cell} python
from petrophys.evaluation import dykstra_parsons, lorenz_coefficient

k = np.asarray(well.get("KH_CORE").data)[core]
phi = np.asarray(well.get("PHIC").data)[core]
dp = dykstra_parsons(k)
print(f"Lorenz {lorenz_coefficient(k, phi).coefficient:.2f}; Dykstra-Parsons V {dp.coefficient:.2f} "
      f"(k50 {dp.k50:.1f} mD, log-normal fit R2 {dp.r_squared:.2f}) from {dp.n_samples} plugs")
```

## Change-point zonation

PELT finds where the mean of one or more curves shifts. The default penalty
finds bed-scale changes; `max_zones` keeps the strongest boundaries. Compare
the result with the formation tops.

```{code-cell} python
from petrophys.evaluation import zone_well

zones = zone_well(well, ["GR", "RHOB"], min_thickness=2.0, top=2700, bottom=3600, max_zones=12)
print(zones.to_dataframe()[["top", "bottom", "GR_mean", "RHOB_mean"]].round(2).to_string(index=False))
```

## Statistics by zone

`zone_statistics` tabulates any curves over any zones: a tops file, a table,
or a list of intervals.

```{code-cell} python
from petrophys.evaluation.summaries import zone_statistics

stats = zone_statistics(well, pp.sample("formations_30_6-5"), ["PHIE", "SW", "VCL"], rank="formation")
brent_stats = stats[stats.zone.isin(["Ness Fm.", "Etive Fm."])]
print(brent_stats[["zone", "curve", "coverage", "arithmetic_mean", "p10", "p50", "p90"]].round(3).to_string(index=False))
```

## Monte Carlo uncertainty

Each uncertain input gets a distribution; the calculation runs on every
draw. Here mean Archie Sw of the Etive net reservoir with the operator's
constants and assumed spreads: Rw within 20 %, m and n between 1.8 and 2.2, and a porosity
calibration error of 0.015 that is the same at every depth
(`correlated_depths`).

```{code-cell} python
from petrophys.evaluation import monte_carlo
from petrophys.interpretation.saturation import archie

etive = (depth >= 2865) & (depth <= 2958) & (np.asarray(well.get("NET_RES").data) > 0.5)
rt = np.asarray(well.get("RDEP").data)[etive]
phie = np.asarray(well.get("PHIE").data)[etive]
rw = float(np.nanmedian(np.asarray(well.get("RW").data)[etive]))

def mean_sw(rt, phi, rw, m, n):
    sw = archie(rt=rt, rw=rw, phi=np.clip(phi, 0.01, None), a=0.8, m=m, n=n)
    return np.nanmean(np.clip(sw, 0, 1), axis=-1)

mc = monte_carlo(mean_sw, {
    "rt": rt, "phi": {"dist": "normal", "mean": phie, "sd": 0.015},
    "rw": ("triangular", 0.8 * rw, rw, 1.2 * rw), "m": ("uniform", 1.8, 2.2), "n": ("uniform", 1.8, 2.2),
}, n=2000, correlated_depths=["phi"])
print("Etive net reservoir, mean Sw p10/p50/p90:", [round(float(v), 3) for v in mc.percentiles.values()])
print("Rank correlation with the answer:", {k: round(v, 2) for k, v in mc.sensitivity.items()})
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
section.

```python
from petrophys.interpretation.pore_pressure import eaton_from_well, fit_normal_compaction_trend

well = pp.read("my_well.las", survey="survey.csv")
nct = fit_normal_compaction_trend(tvd, dtc, kind="sonic", mask=shale & normally_pressured, mudline=135)
ppr = eaton_from_well(well, nct, water_depth=110, density_above_log=1.95)  # air gap from the header
exponent, misfit = ppr.calibrate(mdt_tvdss, mdt_bar, datum="TVDSS")      # fit Eaton's exponent
ppr = eaton_from_well(well, nct, water_depth=110, density_above_log=1.95, exponent=exponent, add_curves=True)
```

Offshore the air gap is the depth reference's elevation above sea level;
onshore it is EKB - EGL, and hydrostatic pressure starts at ground level.
Calibrate Eaton's exponent to measured pressures before using the result.
