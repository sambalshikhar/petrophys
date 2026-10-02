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
for message in [str(item.message) for item in caught] + result.warnings:
    print(fill(f"- {message}", 88, subsequent_indent="  "))
```

The first two are Python warnings: 30/6-5 stores core SWC and SOC in percent under a "frac" label, so Petrophys converts them and says so.
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
| **Clay volume**: gamma-ray endpoints (`gr_min`, `gr_max`, default 20 and 100 API) | the `vclay` step in the trace |
| **Porosity and saturation**: core agreement, chosen Sw model | `result.print_trace()`, the `saturation` step |
| **Hole quality** | the `BAD_HOLE` curve next to the pay flags |
| **Cutoffs**: project values instead of defaults | the `net_pay` step parameters |
| **Geology**: formation and lithology files are optional | logs drive the workflow; a formation name never creates logs or mineral fractions |

Do not tune Rw just to get the pay you expect.

## Implausible inputs

{class}`pp.DataQualityWarning <petrophys.exceptions.DataQualityWarning>` flags values a curve's unit cannot explain. Catch it to see what was removed:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", pp.DataQualityWarning)
    checked = pp.interpret(pp.read(pp.sample("15_9-14")))

for item in caught:
    if issubclass(item.category, pp.DataQualityWarning):
        text = f"{item.category.__name__}: {item.message}"
        print(fill(text, 88, subsequent_indent="    "))
print("status:", checked.status)
```

The run continues without those samples, and the message stays in `result.warnings`.

## Rw needs its temperature

Rw is quoted at a temperature. Without `rw_meas_temp_f`, Petrophys assumes 75 degF and records that in the `rw` step:

```{code-cell} python
assumed = pp.interpret(well, rw=0.07)  # no temperature given
rw_step = next(step for step in assumed.steps_executed if step.step == "rw")
print(rw_step.method)
for name, run in {"Rw at 230 degF": result, "Rw at assumed 75 degF": assumed}.items():
    pay = run.summary(top=2841, bottom=2958)["net_thickness"]["net_pay"]
    print(f"{name:22s} Brent net pay {pay:.1f} m")
```

```{code-cell} python
:tags: [remove-input]
import matplotlib.pyplot as plt
from petrophys.visualization import use_plot_style
use_plot_style()

fig, (ax_rw, ax_sw) = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 2]})
for run, label, color in [(result, "Rw 0.07 at 230 degF", "#2a78d6"),
                          (assumed, "Rw 0.07 at 75 degF (assumed)", "#eb6834")]:
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

Most flags sit in the shallow Nordland and Hordaland groups, where claystones read low clay and high porosity.
The Brent Group (`BAAT GP.` in these tops) is the reservoir here, so summarize that window:

```{code-cell} python
result.summary(top=2841, bottom=2958)["net_thickness"]
```

The flags stay simple on purpose: missing VCL counts as zero clay, and neither `BAD_HOLE` nor an external shale label removes pay.
Check flagged pay against the geology, for example the [Sodir description of the Heather Formation](https://factpages.sodir.no/nb-no/strat/pageview/litho/formations/60), a shale.

## The cutoffs actually used

```{code-cell} python
net_pay = next(step for step in result.steps_executed if step.step == "net_pay")
{key: net_pay.parameters[key]
 for key in ["vcl_cutoff", "phie_cutoff", "sw_cutoff", "cutoff_source"]}
```

Pass project values as `vcl_cutoff`, `phie_cutoff` and `sw_cutoff`.

## Marl

Marl is a clay-carbonate mixture, not pure limestone ([SLB definition](https://glossary.slb.com/en/terms/m/marl)).
Petrophys interprets marl with the limestone matrix and the carbonate cutoffs, and warns when a lithology file contains it, as above. Supply project endpoints and cutoffs for marl-rich intervals.

## Known sample limitations

| Sample | Limitation | What Petrophys does |
|---|---|---|
| 30/6-5 SWC, SOC | labelled frac, stored in percent | converts to fractions on reading and warns; the bundled LAS is unchanged. Give an explicit unit for ambiguous low values |
| 30/6-5 header RW | `RW 0.7` ohm.m with no temperature, ten times the 0.07 used here (likely a slipped decimal) | always pass `rw` and `rw_meas_temp_f` |
| 15/9-14 DTE | values fit no sonic unit; meaning unknown | not treated as shear slowness; range QC removes them. Do not rescale |
| Daydream CMR | scalar NMR curves only: no T2 bins or T2LM; `CMRP_3MS` is labelled V/V but holds porosity units | T2LM cannot be computed; vendor KSDR and KCMR are read, not recomputed |
| 30/6-5 tops | groups and formations nest | use `rank="formation"` or `"group"` in `formation_summary`; core permeability is calibrated per group and per formation |
| COSTA HW-* | each LAS covers only the Kharaib-2 interval | `formations_costa` labels it `KHARAIB-2`; not for whole-well or multi-formation studies |

## DLIS files

The Daydream scalar and borehole-image DLIS files live in the repository with Git LFS, not in the installed package. From a checkout:

```bash
git lfs install
git lfs pull
python -m pip install -e ".[dlis,tutorials]"
```

Tutorial 16 needs the real image DLIS, not the LFS pointer. The inventory and checksums are in `src/petrophys/datasets/daydream/README.md`.

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
