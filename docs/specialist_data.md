---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Specialist data

Logs are rarely the only measurements in a well. NMR, spectral gamma, core, saturation-height and borehole-image data each have their own API and displays; their products join the same `Well` that you interpret and plot.
`pp.interpret` does not run spectral gamma or elemental spectroscopy; call those functions yourself.

```{code-cell} python
:tags: [remove-cell]

import warnings

# Warnings as one line, without the source file.
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
# 30/6-5 labels its percent core saturations (SWC, SOC) as fractions; Petrophys
# converts them to v/v and warns. The page explains this once, in the text.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## NMR: Daydream CMR

The bundled Daydream 1 file is a vendor CMR summary, with GR and calipers.
{func}`~petrophys.visualization.create_nmr_panel` draws the standard display from its curves.

```{code-cell} python
import petrophys as pp
from petrophys.visualization import create_nmr_panel

cmr = pp.read(pp.sample("daydream1_cmr"))
create_nmr_panel(cmr, depth_range=(3900, 4100));
```

{class}`~petrophys.interpretation.nmr.NMRData` loads the same file to compute bound fluid (`MPHI - MFFI`), irreducible saturation and permeability.
The default T2 cutoff is 33 ms, the sandstone value (92 ms for carbonates; Coates, Xiao and Prammer, 1999). With T2 bins, the bin that straddles the cutoff counts with the part of its log-T2 width below it; `partial_bins=False` counts whole bins instead, as some vendor summaries do.

```{code-cell} python
from petrophys.interpretation.nmr import NMRData

nmr = NMRData.from_las(pp.sample("daydream1_cmr"), mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
nmr
```

`nmr.to_dataframe()` holds the curves, the computed ones included: total and
free-fluid porosity (`MPHI`, `MFFI`, in porosity units), bound fluid `MBVI`,
irreducible saturation `SWIRR` and the vendor's Timur-Coates permeability `KTIM`:

```{code-cell} python
nmr.to_dataframe()[["MPHI", "MFFI", "MBVI", "SWIRR", "KTIM"]].describe()
```

`permeability_coates()` and `permeability_sdr()` return the vendor curve unless `prefer_stored=False`, which computes Timur-Coates with the textbook constants. Over this file the two agree closely (median 0.0012 against 0.0013 mD):

```{code-cell} python
import pandas as pd

pd.DataFrame({"Petrophys": nmr.permeability_coates(prefer_stored=False),
              "vendor KTIM": nmr.stored_permeability["KTIM"]}).describe()
```

This file has no T2 bins, so T2 log-mean and SDR from scratch need a richer delivery. Petrophys does not rebuild distributions from raw echo trains.

## Spectral gamma: 25/5-1

The trace confirms that `pp.interpret` skips it:

```{code-cell} python
well25 = pp.read(pp.sample("25_5-1"))
pp.interpret(well25).specific_steps_skipped["spectral_gamma"]
```

{func}`~petrophys.interpretation.geochemistry.interpret_spectral_gamma` finds thorium and uranium, converts units, and adds uranium's share of the gamma ray to the well. The GR mnemonic does not name the spectral tool, so we pass it as the total.

```{code-cell} python
from petrophys.interpretation.geochemistry import interpret_spectral_gamma

gamma = interpret_spectral_gamma(well25, alias_overrides={"SGR": ["GR"]})
gamma.output_curves
```

The run ends at 3432 m with held readings (values repeated exactly, then zeros), so the plot stops at 3423 m. Between 2906 and 3423 m, 371 of the 3395 logged samples get at least half their gamma ray from uranium (`SGR_URANIUM_FRACTION` of 0.5 or more, flagged in `SGR_URANIUM_EXCESS_FLAG`); 350 of them lie in the uranium-rich shale above 2990 m, where uranium reaches 34 ppm:

```{code-cell} python
pp.plot(well25, curves=["GR", "THOR", "URAN", "SGR_URANIUM_FRACTION"], top=2900, bottom=3423,
        curve_styles={"GR": {"right": 320}, "URAN": {"right": 35},
                      "SGR_URANIUM_FRACTION": {"left": 0, "right": 1}})
```

Where uranium dominates, GR over-reads clay and thorium is the safer clay indicator. The thorium-uranium ratio points to the depositional setting: Th/U below 2 suggests marine, reducing conditions, 2 to 7 marine, above 7 continental (`U_TH_RATIO` holds its inverse, U/Th). {func}`~petrophys.interpretation.geochemistry.interpret_elemental_spectroscopy` handles elemental yields, also outside `pp.interpret`.

## Core data and calibration: 30/6-5

The 30/6-5 LAS contains core porosity, permeability, grain-density and saturation curves. {func}`~petrophys.interpretation.core_analysis.core_data_from_well` reads those embedded measurements directly. The LAS labels its percent core SWC and SOC as fractions; Petrophys detects and converts them to fractions with a warning, hidden on this page.

```{code-cell} python
from petrophys.interpretation.core_analysis import (
    calibrate_porosity, core_data_from_well, fit_poro_perm_transform,
)

result = pp.interpret(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"),
                      rw=0.07, rw_meas_temp="230 degF")
core = core_data_from_well(result)
core
```

Compare a log porosity with the plugs, like with like: plug porosity is total
porosity, so the comparison takes `PHID`, the total density porosity of the
interpretation, which uses a quartz matrix and is independent of the core.
(Effective porosity, `PHIE` or `PHIE_CC`, excludes clay-bound water and reads below
the plugs in shaly rock.)

```{code-cell} python
from petrophys.visualization import plot_calibration, plot_core_on_logs, plot_poro_perm

calibration = calibrate_porosity(result, core, log_curve="PHID")
plot_calibration(calibration, title="Density porosity against core");
```

Density porosity reads 0.02 v/v below the plugs on average (RMSE 0.046 v/v, R² 0.65
over 200 plugs). A poro-perm transform fitted to the plugs predicts permeability
from porosity:

```{code-cell} python
transform = fit_poro_perm_transform(core)
plot_poro_perm(core, transform=transform, title="Core poro-perm and transform");
```

`pp.interpret` applies a core calibration itself when the well has core: it writes
core-corrected effective porosity `PHIE_CC` and, from it, permeability `PERM_CC` on
the plugs' scale. Both are written over the whole well: with formation tops, as here,
a formation with at least three plugs (the Brent Group, 2841-2958 m) gets a model of
its own and the other formations the whole-well model, far from the rock the plugs
sample. `PERM_CC` is written only where clay volume is at or below the net-rock clay
cutoff (0.40 for sandstone); `PERM` holds elsewhere. Plot density
porosity and `PERM_CC` with the plugs
({func}`~petrophys.interpretation.core_analysis.apply_poro_perm_transform` applies a
transform of your own to any porosity curve):

```{code-cell} python
plot_core_on_logs(result, core, curves=["PHID", "PERM_CC"], figsize=(8, 7),
                  depth_range=(core.top - 10, core.bottom + 10));
```

Core saturation stays calibration evidence; it does not replace the log Sw curve. Check depth ties first: on 25/5-1 the plugs need a depth shift before they match the logs ({doc}`tutorial 12 <tutorials/part_b/12_core_calibration_and_digital_core>`).

## Saturation-height from core: COSTA

A saturation-height model predicts Sw from height above the free-water level (FWL).
The core-first workflow builds it from laboratory capillary pressure and then uses
the logs only to place the FWL. The bundled COSTA mercury-injection data hold 110
plugs, each with its porosity and permeability; no well or depth is recorded for them:

```{code-cell} python
from petrophys.interpretation.saturation.height import (
    estimate_fwl_from_logs, fit_j_function_from_core, read_capillary_pressure,
    water_contact_from_fwl,
)

cap = read_capillary_pressure(pp.sample("costa_core_pc"))
cap
```

A Thomeer hyperbola fitted to each plug's curve gives its entry pressure `Pd` (psi,
air-mercury), pore geometry factor `G` and mercury volume at infinite pressure
`BV_INF` (percent of bulk volume); the medians over the plugs:

```{code-cell} python
cap.thomeer().median()
```

The Leverett J-function makes the plugs comparable: it scales capillary pressure by
the square root of permeability over porosity. `fit_j_function_from_core` converts the
laboratory curves to oil-brine at reservoir conditions and fits Sw against J over every
point. The fluid densities are assumed, not measured: brine 1.14 g/cm3 (a saline
formation water, about 200,000 ppm) and oil 0.75 g/cm3.

```{code-cell} python
model = fit_j_function_from_core(cap, "oil_brine", sg_water=1.14, sg_hc=0.75)
model
```

```{code-cell} python
model.plot();
```

Sw = 0.172 J^-0.653 fits the 4562 core points with R² 0.68; the scatter is plug-to-plug
pore-geometry variation that one J-function averages. With the model fixed, the FWL
is the depth at which the model, applied to a well's porosity and permeability, best
matches its log Sw. On HW-24, which logs a transition zone:

```{code-cell} python
fit = estimate_fwl_from_logs(pp.read(pp.sample("HW-24")), model, sw_curve="SW_INPUT",
                             perm_curve="PERM_CC", phi_curve="PHIE_CC")
fit
```

```{code-cell} python
fit.plot();
```

The FWL is 8585.1 ft (95 % range 8574.0-8604.1 ft), with an RMS Sw misfit of 0.084:
the model follows log Sw from the low-saturation column down through the transition
zone. HW-24 has no TVD curve, so its depth index is taken as true vertical depth, which
holds for a vertical well only; for a deviated well, pass `tvd_curve`. Where the misfit
keeps falling to the deep end of the search, the result is only a bound: for HW-3, whose
log Sw stays low until the last 10 ft of the file, the answer is "deeper than 8116.7 ft".

The water contact, where Sw first falls below 1, lies above the FWL by the height of the
entry pressure. With the median oil-brine entry pressure of the plugs (1.54 psi), it is
at 8576.0 ft:

```{code-cell} python
entry = cap.to_reservoir("oil_brine").entry_pressure().median()
water_contact_from_fwl(fit.fwl, entry, sg_water=1.14, sg_hc=0.75)
```

`saturation_height_from_logs(well, model, fwl_depth=...)` applies the model along a
well. Without core capillary pressure, `fit_j_function_from_well` fits the J-function
to log Sw above an FWL you give.

## Borehole images

Petrophys orients, equalizes, measures coverage of and draws image logs; it does not pick dips or fractures. This cell needs `petrophys[dlis]` and a 112 MB image file that `integration_fixture` downloads on first use, so it does not run here.

```{code-cell} python
:tags: [skip-execution]

from petrophys.datasets import integration_fixture
from petrophys.interpretation.borehole_imaging import read_fmi_dlis
from petrophys.visualization import create_fmi_panel

image = read_fmi_dlis(integration_fixture("daydream1_fmi_processed_dynamic_image.dlis"))
image
```

```{code-cell} python
:tags: [skip-execution]

create_fmi_panel(image, depth_range=(4052, 4058), orient=True, equalize="dynamic")
```

`orient_to_azimuth` turns an image to north, `azimuthal_coverage` gives the imaged
fraction of the circumference and `plot_fmi` draws an image with static or dynamic
equalization, all in `petrophys.interpretation.borehole_imaging`; cut a depth window
with `image.filter_by_depth(top, bottom)` before orienting.
{doc}`Tutorial 16 <tutorials/part_b/16_borehole_image_fmi>` runs them on the real Daydream 1 FMI.

## Core-scanner, NMR and capillary-pressure data

The EpsLog EDE-01 collection includes core-scanner measurements and matching
composite well logs from NLOG.NL (`petrophys.list_samples("epslog")`). Its
README records the curve mappings, units and depth-matching limits.

| Data | API | Bundled data | Not included | Tutorial |
| --- | --- | --- | --- | --- |
| XRF concentrations and mineral proxies | `XRFData`, `create_xrf_panel` | EDE-01 concentrations and matching well logs | XRD mineral fractions | 12 |
| Grain size and permeability comparison | `GrainSizeData` | EDE-01 merged P50 and probe permeability | P13/P87 and measured core porosity for Kozeny-Carman calibration | 12 |
| Scanner strength and dynamic mechanics | `calculate_elastic_properties` | EDE-01 scanner strength and density/sonic well logs | core VP/VS LAS, static moduli and plug UCS tests | 12 |
| NMR T2 distributions and SDR | `NMRData` with bins | Phillips teaching example | a field T2 distribution with matching core | 14 |
| Laboratory capillary pressure | `read_capillary_pressure` (`.thomeer()`, `.to_reservoir()`, `.entry_pressure()`), `fit_j_function_from_core`, `estimate_fwl_from_logs`, `water_contact_from_fwl` | COSTA mercury-injection curves | fluid properties and the well and depth of each plug | 15 |

`read_formation_pressure` accepts CSV/LAS station files, DataFrames and LAS objects.
Fluid labels are optional. For unlabeled stations call `.infer_contacts()` and
`create_pressure_contact_plot(data, result)`. This compares up to three robust
depth-ordered gradients, estimates fluid densities, evaluates possible fluid
sequences and intersects adjacent lines. It returns model scores, assignments,
suspect flags and conditional sensitivity intervals. A single gradient, weak
separation, competing models or ambiguous fluid identities do not yield a definite
FWL. Density bounds and pressure error are configurable interpretation assumptions.
Do not use a selection mask drawn from known fluid intervals as measurement QC.
The plot frames the stations assigned to a gradient and notes how many lie
outside; `zoom="all"` shows every station.

The bundled station files carry measurements only, without fluid labels. In
Tutorial 15, 35/11-13 uses the 21 tests with `QC` 0, including the shallow gas
points, and gives an oil/water FWL near 3113.2 m (report: 3112.6 m TVD MSL); the
two-point gas leg is not resolved separately. For 35/11-10 the tutorial selects the
static pretests of the Fensfjord interval (2250-2352 m TVD MSL): the oil/water
candidate is near 2290.7 m, with ambiguous fluid
hypotheses, so `fwl_depth` is `None`. Report contacts are used only for external
comparison. These are limited reproduction checks, not validation across unseen
reservoirs.

Raw MDT pretest picking works on time-indexed DLIS, LAS, CSV or DataFrame
inputs through `read_mdt_pretest`. Three original MDT pretest DLIS files from
IODP Hole C0009A are bundled with LAS copies (`319-C0009A_MDT_059LTP` and
`c0009a_mdt_059`, likewise 067 and 078). `pick_pressure()` accepts 4102.13 psi
for file 059; the low-mobility test 067 and the unstable-baseline test 078 are
returned for review. `create_mdt_pretest_plot(trace, pick)` shades the mud
baselines at both ends and the buildup windows, with the baseline that failed the
stability check in red. DLIS input needs the `dlis` extra. After a user
supplies verified TVD values, `formation_pressures_from_mdt` builds station
data for `.infer_contacts()`.

For already interpreted fluid labels, `.fit_fwl()`, `fit_pressure_fwl` and
`create_formation_pressure_plot` are available. Images and report contacts
are not interpretation inputs. Units, a positive-down TVD datum, a consistent
pressure reference and reservoir selection are necessary for either mode.

Pressure alone does not determine a saturation-height function. Capillary-pressure
or rock-property calibration and fluid properties are still required.
`saturation_height_from_logs` takes `fwl_depth` in the unit of the well's depth
index, or of `tvd_curve` when one is given, and on the same datum; it converts
heights in metres to feet for the capillary relation itself. Neither of these
contacts applies to the separate COSTA example.

Sonic saturation (`petrophys.interpretation.saturation.sonic`) estimates Sw without resistivity, by Gassmann substitution, and says when the fluid effect is undetectable.

## Next

{doc}`analysis_tools`: the analyses `pp.interpret` does not run, from clay typing to pore pressure.

The tutorials go deeper on {doc}`core calibration <tutorials/part_b/12_core_calibration_and_digital_core>`, {doc}`NMR <tutorials/part_b/14_nmr_petrophysics>`, {doc}`saturation-height <tutorials/part_b/15_capillary_pressure_saturation_height>` and {doc}`borehole images <tutorials/part_b/16_borehole_image_fmi>`.
