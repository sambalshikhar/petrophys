---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Reviewing an interpretation

A completed run shows that the calculations ran; it does not make the numbers a calibrated interpretation.
This page runs the checks worth making before you use a result, on the bundled sample wells.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full
```

## Read the warnings first

```{code-cell} python
import warnings
from textwrap import fill

import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"),
               lithology=pp.sample("lithology_30_6-5"))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)  # Rw 0.07 ohm.m at 230 degF

print("status:", result.status)
# Print each message once: the core-saturation notes are raised many times.
for message in dict.fromkeys([str(item.message) for item in caught] + result.warnings):
    print(fill(f"- {message}", 88, subsequent_indent="  "))
```

The first two are Python warnings: 30/6-5 stores core SWC and SOC in percent under a "frac" label, so the curve standardization converts them to v/v and says so.
The rest stay on the result in `result.warnings`. From here on, this page hides only the SWC/SOC message.

```{code-cell} python
:tags: [hide-cell]
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## A review checklist

| Check | Where to look |
|---|---|
| **Inputs**: expected curves, right units | `well.describe()`; `DataQualityWarning` messages |
| **Rw and its temperature** | pass `rw_meas_temp_f` with `rw`; the `rw` step in the trace |
| **Clay volume**: gamma-ray endpoints (`gr_min`, `gr_max`; by default P5 and P95 of the log, with the clean endpoint at most 20 API and the shale endpoint at least 100 API) and the neutron-density clay point (`nd_clay`) | the `vclay` step in the trace |
| **Porosity and saturation**: core agreement; the core porosity correction, used only when its fit has at least 8 core-log pairs, a positive slope and r² of at least 0.5; the chosen Sw model | `result.print_trace()`, the `core_correction` and `saturation` steps |
| **Hole quality**: bad-hole samples are left out of net reservoir and pay; where no bit size is known, only DRHO can flag them | the `BAD_HOLE` curve next to the pay flags; the warnings and sample counts of the `bad_hole` step |
| **Cutoffs**: project values instead of the lithology-family standards | the `net_pay` step parameters |
| **Geology**: formation and lithology files are optional | logs drive the workflow; a formation name never creates logs or mineral fractions |
| **Calculation profile**: the interpretation configuration selected for your task | `result.config.calculation_profile`; see {doc}`workflows` |

Do not tune Rw just to get the pay you expect.

## Implausible inputs

{class}`DataQualityWarning <petrophys.exceptions.DataQualityWarning>` flags values a curve's unit cannot explain. Catch it to see what was removed:

```{code-cell} python
from petrophys.exceptions import DataQualityWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DataQualityWarning)
    checked = pp.interpret(pp.read(pp.sample("15_9-14")))

for item in caught:
    if issubclass(item.category, DataQualityWarning):
        text = f"{item.category.__name__}: {item.message}"
        print(fill(text, 88, subsequent_indent="    "))
print("status:", checked.status)
```

The run continues without those samples, and the message stays in `result.warnings`.

`pp.interpret` raises the lower limit of the DTC range and adds one for SP. Compressional slowness (DTC) below 38.5 us/ft is removed: no rock matrix is faster (dolomite is 38.5-43.5 us/ft, calcite 43.5-47.6 and quartz 51-55.5; Schlumberger, 1989), so a lower value is a unit or scale error.
SP beyond ±500 mV is removed: a formation's static SP lies within about 200 mV of the shale baseline, so such readings come from the tool or the recording.
The workflow also checks these inputs:

| Input | What Petrophys does |
|---|---|
| Header bit size outside 3-36 in (a null such as -999.25, or a value in mm) | Ignored, with a warning |
| No bit size at a sample, from the BS curve or a 3-36 in header value | No size is assumed: the caliper washout test (CALI more than 1.5 in over the bit size) and the large-hole test (bit size above 17.5 in) do not run there, so `BAD_HOLE` there comes from an absolute DRHO above 0.15 g/cm3 alone. A warning names the depths where a caliper reading has no bit size |
| A caliper, but neither a bit size nor DRHO | The `bad_hole` step is skipped, with the reason in `result.specific_steps_skipped`, and no `BAD_HOLE` curve is written |
| Logs that read like halite where no lithology code says evaporite: RHOB 1.95-2.15 g/cm3, GR below 15 API, RT above 500 ohm.m (or, without RT, a caliper more than 0.5 in under the bit size), and NPHI at most 0.05 and PEF 4.0-5.5 where logged | Porosity uses the halite endpoints; the samples are left out of net reservoir and pay |
| A neutron-density clay point less than 0.10 v/v of neutron above the clean line | VCL_ND is reported but left out of VCL, with a warning |
| Neutron readings beyond both the clean line's 40 p.u. end point and the clay point, as in undercompacted mudstone | VCL_ND is left undefined there, and VCL comes from the other indicators |

`BAD_HOLE` = 1 marks washout or large hole, where density and neutron readings, and the PHIE and SW built on them, are unreliable.
Each sample's bit size is the BS curve, else a 3-36 in header value, which also fills gaps in the BS curve.
An absolute DRHO above 0.15 g/cm3 flags washout wherever DRHO is logged, also in a well without a caliper.
The `bad_hole` step records where its bit sizes came from (`bit_size_source`) and counts the samples without a bit size (`no_bit_size_samples`), the samples given the caliper test (`caliper_test_samples`) and the samples neither test reaches (`not_assessed_samples`), where `BAD_HOLE` stays 0.

## Rw needs its temperature

Rw is quoted at a temperature. Without `rw_meas_temp_f`, Petrophys takes Rw at the well's median formation temperature, records that in the `rw` step and says so in `result.warnings`:

```{code-cell} python
assumed = pp.interpret(well, rw=0.07)  # no temperature given
rw_step = next(step for step in assumed.steps_executed if step.step == "rw")
print(rw_step.method)
print(fill(next(text for text in assumed.warnings if "rw_meas_temp_f" in text), 88))
for name, run in {"Rw at 230 degF": result, "Rw at the median temperature": assumed}.items():
    pay = run.summary(top=2841, bottom=2958)["net_thickness"]["net_pay"]
    print(f"{name:30s} Brent net pay {pay:.1f} m")
```

```{code-cell} python
:tags: [remove-input]
import matplotlib.pyplot as plt
from petrophys.visualization import use_plot_style
use_plot_style()

fig, (ax_rw, ax_sw) = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 2]})
for run, label, color in [(result, "Rw 0.07 at 230 degF", "#2a78d6"),
                          (assumed, "Rw 0.07 at the median temperature (assumed)", "#eb6834")]:
    logs = run.well.to_dataframe()
    brent = logs[(logs["DEPTH"] >= 2841) & (logs["DEPTH"] <= 2958)]
    ax_rw.plot(brent["RW"], brent["DEPTH"], color=color, lw=1.5, label=label)
    ax_sw.plot(brent["SW"], brent["DEPTH"], color=color, lw=1, label=label)
ax_rw.set_xlabel("Rw at formation temp. (ohm.m)")
ax_rw.set_xlim(0, None)
ax_sw.set_xlabel("SW (v/v)")
ax_sw.set_xlim(0, 1)
ax_rw.set_ylabel("depth (m)")
ax_rw.invert_yaxis()
ax_sw.legend(loc="lower left", fontsize=8, framealpha=0.9)
for axis in (ax_rw, ax_sw):
    axis.grid(alpha=0.3)
fig.suptitle("30/6-5, Brent: the same Rw quoted at two temperatures", fontsize=11)
fig.tight_layout()
plt.show()
```

## Where the pay sits

`NET_ROCK`, `NET_RES` and `NET_PAY` are cutoff flags, not confirmed hydrocarbons.
Summarize them per group before trusting a total:

```{code-cell} python
groups = result.formation_summary(rank="group")
columns = ["Formation", "Top", "Bottom", "Net Pay (m)", "Avg PHIE", "Avg Sw"]
groups[columns].style.hide(axis="index").format(precision=2, na_rep="–")
```

```{code-cell} python
:tags: [remove-input]
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [f"{name}  {top:.0f}-{bottom:.0f} m" for name, top, bottom in
          zip(groups["Formation"], groups["Top"], groups["Bottom"])]
colors = ["#087f73" if name == "BAAT GP." else "#b8c0c7" for name in groups["Formation"]]
bars = ax.barh(labels, groups["Net Pay (m)"], color=colors, height=0.7)
ax.bar_label(bars, fmt="%.0f m", padding=3, fontsize=8, color="#3b4550")
ax.invert_yaxis()
ax.set_xlim(0, 1.12 * groups["Net Pay (m)"].max())
ax.set_xlabel("flagged net pay (m)")
ax.grid(axis="x", alpha=0.3)
ax.set_title("30/6-5: net-pay flags by group, default cutoffs (Brent Group in teal)", fontsize=11)
fig.tight_layout()
plt.show()
```

With these tops and lithology attached, nearly all the flagged pay lies in the Brent Group (`BAAT GP.` in these tops), the reservoir. The 1.8 m flagged elsewhere, in the Rogaland and Shetland groups, still needs a geological check.
Summarize the reservoir window:

```{code-cell} python
result.summary(top=2841, bottom=2958)["net_thickness"]
```

The flags are cutoff screens.
Evaporite, coal, halite recognized on the logs and bad-hole (`BAD_HOLE` = 1) samples are left out of net reservoir and net pay, and the `net_pay` step records how much each removed (below).
An external shale label does not remove pay by itself, and a well with no clay-volume curve at all is screened as clean rock (VCL = 0), with a warning.
Check flagged pay against the geology, for example the [Sodir description of the Heather Formation](https://factpages.sodir.no/nb-no/strat/pageview/litho/formations/60), a shale.

:::{admonition} Known limitation: one Rw for the whole well
:class: warning

A run uses one formation-water resistivity for the whole well, corrected for temperature at constant salinity, but shallow formation waters are usually fresher than the reservoir's.
In undercompacted mudstones the clay indicators also disagree, so clay volume can read low.
Where no lithology code covers a shallow Tertiary section, a whole-well run can therefore still flag marginal pay there, at SW about 0.5.
On 25/5-1, for example, a run with default settings flags 259 m of pay above the reservoir top at 2984 m, almost all of it in uncoded Tertiary rock.

Read the results per formation (attach tops), or give the interval its own Rw: the zoned runner takes `RW_Z1` between `TOP_Z1` and `BOT_Z1` from the LAS header ({doc}`workflows`), or interpret the interval on its own with its `rw`.
Petrophys does not yet model a formation-water salinity that changes with depth.
:::

## The cutoffs actually used

Cutoffs you leave unset take the standard of each sample's lithology family (`STANDARD_CUTOFFS` in `petrophys.evaluation`): clastic VCL < 0.40, PHIE > 0.08 and SW < 0.60; carbonate (limestone, chalk, dolomite and marl) VCL < 0.30, PHIE > 0.045 and SW < 0.70.
The 30/6-5 lithology file has samples of both families, so the step lists the cutoffs per family:

```{code-cell} python
net_pay = next(step for step in result.steps_executed if step.step == "net_pay")
print(fill(f"cutoff_source: {net_pay.parameters['cutoff_source']}", 88, subsequent_indent="    "))
pd.DataFrame.from_dict(net_pay.parameters["cutoffs_by_family"], orient="index")
```

The samples left out of net reservoir and pay, and the thickness each kind removed over the whole well. On 30/6-5 only bad hole removes any:

```{code-cell} python
excluded = net_pay.parameters["excluded_from_net_reservoir_and_pay"]
pd.DataFrame.from_dict(excluded, orient="index")
```

Pass project values as `vcl_cutoff`, `phie_cutoff` and `sw_cutoff`; a value you set applies to every sample.

## Marl

Marl is a clay-carbonate mixture, not pure limestone ([SLB definition](https://glossary.slb.com/en/terms/m/marl)).
Petrophys interprets marl with the limestone matrix and the carbonate cutoffs, and warns when a lithology file contains it, as above. Supply project endpoints and cutoffs for marl-rich intervals.

## Known sample limitations

| Sample | Limitation | What Petrophys does |
|---|---|---|
| 30/6-5 SWC, SOC | labelled frac, stored in percent | `standardize_curves` (and so `interpret`) converts them to v/v with the warnings shown above; the raw-well plots convert for display only; the bundled LAS is unchanged. Give an explicit unit for ambiguous low values |
| 30/6-5 header RW | `RW 0.7` ohm.m with no temperature, ten times the 0.07 used here (likely a slipped decimal) | always pass `rw` and `rw_meas_temp_f` |
| 15/9-14 DTE | values fit no sonic unit; meaning unknown | not treated as shear slowness; range QC removes them. Do not rescale |
| Daydream CMR | scalar NMR curves only: no T2 bins or T2LM; `CMRP_3MS` is labelled V/V but holds porosity units | T2LM cannot be computed; vendor KSDR and KCMR are read, not recomputed |
| 30/6-5 tops | groups and formations nest | use `rank="formation"` or `"group"` in `formation_summary`; core permeability is calibrated per group and per formation |
| COSTA HW-* | each LAS covers only the Kharaib-2 interval | `formations_costa` labels it `KHARAIB-2`; not for whole-well or multi-formation studies |

## DLIS files

The Daydream scalar and borehole-image DLIS files are too large for the installed package. `integration_fixture` downloads each one on first use, checks it against its recorded SHA-256 and keeps it in `petrophys.datasets.data_dir()` (set `PETROPHYS_DATA_DIR` to choose the folder):

```python
from petrophys.datasets import integration_fixture

path = integration_fixture("daydream1_fmi_processed_dynamic_image.dlis")
```

Reading DLIS needs `pip install "petrophys[dlis]"`. A source checkout with `git lfs pull` uses its own copies instead. The inventory and checksums are in `src/petrophys/datasets/daydream/README.md`.

## Go deeper

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} QC and standardization
:link: tutorials/part_a/02_qaqc_and_standardization
:link-type: doc
Units, ranges, aliases and bad-hole flags.
:::

:::{grid-item-card} Formation water and temperature
:link: tutorials/part_a/06_formation_water_and_environment
:link-type: doc
Where Rw comes from and how temperature changes it.
:::

:::{grid-item-card} Reservoir quality and net pay
:link: tutorials/part_a/09_reservoir_quality_and_net_pay
:link-type: doc
Cutoffs, flags and what they do not prove.
:::

:::{grid-item-card} Saturation model sensitivity
:link: tutorials/case_studies/cs5_saturation_model_sensitivity_30_6-5
:link-type: doc
How much the Sw model changes the Brent answer.
:::
::::
