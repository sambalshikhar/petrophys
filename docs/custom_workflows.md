---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

(custom-workflows)=
# Custom workflows

`pp.interpret` chooses the methods from the data ({doc}`workflows`). When a project needs a fixed recipe, build your own from built-in steps and your own functions.
Check it against a well, run it on one well or a whole folder, and save it to share.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import warnings

import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full

# 30/6-5 stores core saturation in percent under a "frac" label. pp.interpret
# converts it for its core calibration and warns (explained below). Only that
# message is hidden.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## Build a workflow

Add steps in the order they run. Each step states the curves it needs and the curves it adds.

```{code-cell} python
import numpy as np
import petrophys as pp
from petrophys.workflow import Workflow, WorkflowValidationError, steps, workflow_step

wf = Workflow("Brent Group quick look")
wf.add(steps.clay_volume(method="gr", gr_clean=20, gr_shale=110))
wf.add(steps.porosity(method="density", matrix_density=2.65))
wf.add(steps.formation_water(rw=0.07, rw_temp_f=230))   # Rw 0.07 ohm.m at 230 degF
wf.add(steps.saturation(model="archie", a=1, m=2, n=2))
wf.add(steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.10, sw_cutoff=0.60))
wf.describe()
```

Parameters are checked when a step is created, so a typo fails at once and names the closest valid option.

```{code-cell} python
from textwrap import fill

def show(error):
    """Print an error message wrapped to the page width."""
    for line in str(error).splitlines():
        print(fill(line, 88, subsequent_indent="      "))

try:
    steps.porosity(method="densty")
except ValueError as error:
    show(error)
```

## Check a well before running

`check` lists every input of every step, counting the curves that earlier steps add.
Here the Rw step is left out on purpose.
(`pp.interpret` warns that this well's core saturation is stored in percent, and converts it for its core calibration; this page hides that one notice.)

```{code-cell} python
well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
draft = wf.copy("draft").remove("formation_water")
report = draft.check(well)
report[~report["Met"]].style.hide(axis="index")
```

A run is strict by default: it stops before any calculation and lists every unmet input.
With `strict=False` it runs the other steps and records why each skipped step could not run.

```{code-cell} python
try:
    draft.run(well)
except WorkflowValidationError as error:
    show(error)
```

## Run and read the trace

`run` works on a copy of the well. Every step records its method, inputs, outputs, parameters and decisions.

```{code-cell} python
run = wf.run(well)
run.to_dataframe()[["Step", "Status", "Method", "Output Curves"]].style.hide(axis="index")
```

```{code-cell} python
brent = run.summary(top=2841, bottom=2958)   # Brent Group window
print("status:", run.status)
print("net pay in the Brent Group:", brent["net_thickness"]["net_pay"], "m")
print({name: round(stats["mean"], 3) for name, stats in brent["key_results"].items()
       if name in ("VCL", "PHIE", "SW")})
```

## Plot the result

The interpreted well plots like any other well.

```{code-cell} python
viewer = pp.plot(run.well, top=2830, bottom=2970, figsize=(11, 10),
                 tracks=["LITHOLOGY", "GAMMA / HOLE", "RESISTIVITY", "POROSITY",
                         "SATURATION", "FLAGS"])
viewer.show()
viewer.close()
```

## Save and share

A workflow saves as a short JSON file that anyone can load and run on their wells.
`run.to_las(path)` also writes the workflow into the LAS parameter section.

```{code-cell} python
import json
import pathlib
import tempfile

with tempfile.TemporaryDirectory() as folder:
    path = wf.save(pathlib.Path(folder) / "brent_quicklook.json")
    print(path.name)
    for step in json.loads(path.read_text())["steps"]:
        print(f"  {step['name']}: {step['params']}")
    loaded = Workflow.load(path)
print("loaded == wf:", loaded == wf)
```

## Try a parameter without editing the workflow

`params` changes a step for one run only.

```{code-cell} python
for n in (1.8, 2.0, 2.2):
    trial = wf.run(well, params={"saturation": {"n": n}})
    pay = trial.summary(top=2841, bottom=2958)["net_thickness"]["net_pay"]
    print(f"n = {n}: Brent Group net pay {pay} m")
```

## Add your own step

Decorate a function that takes the well and returns its curves.
`requires` is checked before running; `produces` gives each curve its unit.

```{code-cell} python
@workflow_step(requires=["PHIE", "SW", "NET_PAY"], produces={"PHIH": "v/v"})
def hydrocarbon_porosity(well, pay_only=True):
    """Hydrocarbon-filled porosity, PHIE * (1 - SW)."""
    phih = well.get("PHIE").data * (1 - well.get("SW").data)
    if pay_only:
        phih = np.where(well.get("NET_PAY").data == 1, phih, 0.0)
    return {"PHIH": phih}

wf.add(hydrocarbon_porosity())
run = wf.run(well)
depth = run.well.depth
brent_rows = (depth >= 2841) & (depth <= 2958)
hpt = np.nansum(run.well.get("PHIH").data[brent_rows]) * run.well.step
print(f"Brent Group hydrocarbon pore thickness: {hpt:.1f} m")
```

:::{note}
Saved workflows store a custom step by its import path (`module:function`).
Keep steps you share in a module; a step defined in a notebook reloads only in that session.
:::

## Extend `pp.interpret`

`steps.interpret()` runs the whole prebuilt workflow as one step, so your own steps can follow it.

```{code-cell} python
auto_plus = Workflow("pp.interpret + hydrocarbon porosity", [
    steps.interpret(rw=0.07, rw_meas_temp_f=230),
    hydrocarbon_porosity(),
    steps.formation_summary(rank="formation"),
])
auto = auto_plus.run(well)
table = auto.artifacts["formation_summary"]
brent = table[table["Formation"].isin(["Ness Fm.", "Etive Fm."])]
columns = ["Formation", "Net Pay (m)", "Avg PHIE", "Avg Sw"]
brent[columns].style.hide(axis="index").format(precision=3)
```

## Run many wells

`run_many` takes wells, files or a folder, and records a failing well instead of stopping.
Each COSTA file covers only the Kharaib-2 carbonate interval; `rw="header"` reads each well's own Rw.

```{code-cell} python
carbonate = Workflow("Kharaib-2 screen", [
    steps.temperature(),
    steps.clay_volume(method="gr", gr_clean=10, gr_shale=80),
    steps.porosity(method="neutron_density", matrix="limestone"),
    steps.formation_water(rw="header"),
    steps.saturation(model="archie"),
    steps.net_pay(vcl_cutoff=0.30, phie_cutoff=0.10, sw_cutoff=0.60,
                  lithology="carbonate"),
])
names = ["HW-3", "HW-4", "HW-6", "HW-24", "HW-25", "HW-29"]
batch = carbonate.run_many([pp.sample(name) for name in names])
field = batch.summary_table()
columns = ["Well", "Status", "Gross", "Net Pay", "Pay/Gross", "Avg PHIE", "Avg Sw"]
field[columns].style.hide(axis="index").format(precision=2)
```

```{code-cell} python
:tags: [remove-input]
import matplotlib.pyplot as plt
from petrophys.visualization import use_plot_style
use_plot_style()

ranked = field.sort_values("Net Pay")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(ranked["Well"], ranked["Net Pay"], color="#087f73")
ax.set_xlabel("Net pay (ft)")
ax.set_title("Net pay by well, Kharaib-2 interval (custom workflow)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
```

## Custom workflow or `pp.interpret`?

| | {func}`pp.interpret <petrophys.api.interpret>` | Custom workflow |
|---|---|---|
| Methods | Chosen per well from the available data | Exactly the steps and parameters you list |
| Missing inputs | The step is skipped with a reason | Reported before running; `strict=False` skips |
| Your own calculations | Not supported | Any function with `@workflow_step` |
| Reuse | Settings passed on each call | A JSON file, run on one well, a list or a folder |
| Good for | A first look at an unfamiliar well | Project recipes, field studies, audits |

The two combine: `steps.interpret()` puts the automatic workflow inside your own.

## Built-in steps

| Step | Calculates | Adds |
|---|---|---|
| `standardize` | Standard curve names and units | aliases such as `RDEP`, `RHOB` |
| `temperature` | Formation temperature from BHT or a gradient | `TEMP` |
| `clay_volume` | Clay volume from GR, SP, neutron-density or their minimum | `VCL` |
| `porosity` | Density, neutron-density, neutron or sonic porosity | `PHIT`, `PHIE` |
| `formation_water`, `mud_filtrate` | Rw and Rmf at formation temperature | `RW`, `RMF` |
| `saturation` | Archie or a shaly-sand model | `SW`, `SXO` |
| `permeability` | Timur, Coates, Wyllie-Rose (`k = 62500 PHIE^6 / Swirr^2`, the Morris-Biggs oil constant) or core-calibrated | `PERM` |
| `rock_quality`, `bvw`, `mhi` | RQI and FZI, bulk volume water, moveable hydrocarbons | `RQI`, `FZI`, `BVW`, `MHI` |
| `net_pay` | Cutoff flags, with each sample's lithology-family cutoffs where none is given; evaporite, coal, halite recognized on the logs and bad-hole samples are not net reservoir or pay (`exclude_non_reservoir=False` applies the cutoffs alone) | `NET_ROCK`, `NET_RES`, `NET_PAY` |
| `spectral_gamma` | K-Th-U standardization and ratios | as available |
| `interpret` | The whole of `pp.interpret` | `VCL`, `PHIE`, `SW`, `PERM`, ... |
| `formation_summary` | Per-formation table | `run.artifacts[...]` |

Parameters and defaults are in the API reference: {mod}`petrophys.workflow.steps`.

`steps.net_pay` leaves out the same samples as `pp.interpret`.
{func}`~petrophys.evaluation.net_pay.summarize_pay` and {func}`~petrophys.evaluation.net_pay.calculate_net_pay` recompute the flags from the curves with one cutoff set and no exclusions, so their totals can differ from a run's `NET_PAY`: summarize a run's own flags with `run.summary()` or the `formation_summary` step.

## Next

{doc}`visualization_reporting`: plot the result of any workflow and export it.

{doc}`Tutorial 21 <tutorials/part_b/21_custom_workflows>` builds a clastic and a carbonate workflow with your own steps, and {doc}`tutorial 18 <tutorials/part_b/18_multiwell_field_study>` screens and ranks all seventeen COSTA wells.
