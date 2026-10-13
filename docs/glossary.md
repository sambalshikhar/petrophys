---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Terms, Curves and Steps

A reference for the names in results, traces and settings. The step and curve
tables are generated from the library and a real run on 30/6-5, so they always
match the installed version.

```{code-cell} python
:tags: [remove-cell]
import warnings
import pandas as pd

# Display settings for this page: whole table cells, warnings without file paths.
pd.set_option("display.max_colwidth", None)
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
```

```{code-cell} python
:tags: [hide-input]
import warnings
import petrophys as pp

# The bundled LAS stores its SWC/SOC core saturations as percentages under a 'frac'
# label; Petrophys converts them and warns. Hide only that message.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")

well = pp.read(pp.sample("30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
print("30/6-5 interpreted:", result.status, "|", len(result.steps_executed), "steps ran")
```

## Terms

| Term | Meaning |
| --- | --- |
| Lithology domain | The mineral model for the whole run: `clastic`, `carbonate`, `evaporite` or `auto` (`lithology_domain`). Not a per-depth label. |
| Lithology intervals | Optional per-depth rock labels from a table. They override labels in the LAS and from the mineral solve. |
| Lithology code curve | A per-sample curve of lithology codes in the LAS: the curve named by `lithology_curve`, else `LITHOLOGY_GEOLINK`, else a curve under another lithology-code name such as `LITHOLOGY` or `FACIES`. Its codes are decoded with the GeoLink/Kingdom code table; see {doc}`input_templates`. |
| Formations | Optional named depth intervals (groups, formations, members) for summaries, plots and core calibration. |
| Rw | Formation-water resistivity in ohm.m, at a stated temperature (`rw_meas_temp_f`). The workflow corrects it to formation temperature. |
| Cutoffs | Limits on clay volume, porosity and water saturation (`vcl_cutoff`, `phie_cutoff`, `sw_cutoff`). Left unset, each sample takes the standard of its lithology family, clastic or carbonate. |
| Net rock, reservoir, pay | Samples passing the clay cutoff; then also the porosity cutoff; then also the saturation cutoff. Evaporite, coal, halite recognized on the logs and bad-hole samples are never net reservoir or pay. |
| Bad hole | `BAD_HOLE` = 1: the caliper reads more than 1.5 in over the bit size, the bit is larger than 17.5 in, or the absolute density correction DRHO is above 0.15 g/cm3. The caliper and bit-size tests run only where a bit size is known; see {doc}`data_quality`. |
| Class tables | The classes each property is graded into, such as `CLEAN_FORMATION` for clay volume or `GOOD_RESERVOIR` for reservoir quality, with a `NO_*_DATA` label where a value is missing: `CLASS_TABLES` in `petrophys.model`. Context labels and the diagnostics analyzers use them; see {doc}`context`. |
| Calculation profile | Which calculations a run uses (`calculation_profile`): `"current"` (the default), or `"geolink"`, which keeps the calculations of the GeoLink eval context package; see {ref}`calculation profiles <calculation-profiles>`. |
| Trace | The record of a run: each step's status, method, inputs, outputs, parameters and decisions. |
| Context | A compact JSON summary of an interpreted well for LLMs and other tools; see {doc}`context`. |

## Where each input goes

The same input has a different name in each interface. File layouts are in
{doc}`input_templates`.

| Input | Python | `export_context` | Command line |
| --- | --- | --- | --- |
| Formations | `pp.read(..., formations=)` | `tops_file` | `--formations` |
| Lithology intervals | `pp.read(..., lithology=)` | `lithology_file` | `--lithology` |
| Lithology codebook | `pp.read_lithology_codes` | `lithology_mapping_file` | `--lithology-codes` |

Through MCP, `load_well` takes `formations` and `lithology`, and `interpret_well`
takes the settings; codebooks are not available there.

:::{warning}
In `pp.read`, `lithology=` attaches lithology intervals. In `pp.interpret`,
`lithology=` is the deprecated name of `lithology_domain`; use
`lithology_domain` there.
:::

## Workflow steps

`pp.interpret` runs these steps in order, each only when its inputs exist. The
last two columns show what happened on 30/6-5. Rows marked "Trace only" are
recorded in every trace but are not step names you can skip.

```{code-cell} python
:tags: [hide-input]
from textwrap import shorten
from petrophys.workflow import workflow_steps

trace_only = {
    "input_context": "Records which formation and lithology sources were used",
}
executed = {step.step: step for step in result.steps_executed}
described = [item["name"] for item in workflow_steps()]
order = list(described)
for previous, step in zip([None] + list(executed), executed):
    if step not in order:  # a trace-only step: place it after the step that ran before it
        order.insert(order.index(previous) + 1 if previous else 0, step)

# Shorten the long module path in the "call it yourself" descriptions.
descriptions = {item["name"]: item["description"].replace("petrophys.interpretation.geochemistry.", "")
                for item in workflow_steps()}
rows = []
for name in order:
    ran = name in executed
    outputs = ", ".join(executed[name].output_curves) if ran else ""
    rows.append({
        "Step": name,
        "What it does": descriptions.get(name) or "Trace only: " + trace_only.get(name, "recorded by the workflow"),
        "On 30/6-5": "ran" if ran else "skipped",
        "Curves written": (shorten(outputs, 40, placeholder=" ...") if name != "normalization"
                           else "input curves, standardized"),
    })
pd.DataFrame(rows).style.hide(axis="index")
```

`result.specific_steps_skipped` gives the reason for every skipped step.
`skip_steps=[...]` skips steps on purpose.

## Curves the workflow adds

Every curve `pp.interpret` added on 30/6-5, with the unit and description it
carries and the step that first wrote it. Other inputs can add or omit curves;
fractions are in v/v.

```{code-cell} python
:tags: [hide-input]
import numpy as np

writer = {}
for step in result.steps_executed:
    for name in step.output_curves:
        writer.setdefault(name, step.step)

rows, kept = [], {}
for item in result.well.describe()["curves"]:
    name = item["mnemonic"]
    if name in well.curves:
        continue  # an input curve
    data = result.well[name].data
    twin = next((other for other, (unit, text, values) in kept.items()
                 if (unit, text) == (item["unit"], item["description"]) and np.array_equal(values, data, equal_nan=True)), None)
    if twin:  # the same values under a second, compatibility name
        rows[list(kept).index(twin)]["Curve"] += f" (also {name})"
        continue
    kept[name] = (item["unit"], item["description"], data)
    rows.append({"Curve": name, "Unit": item["unit"], "Description": item["description"],
                 "Written by": writer.get(name, "not listed in the trace")})
pd.DataFrame(rows).style.hide(axis="index")
```

The cutoffs behind `NET_ROCK`, `NET_RES` and `NET_PAY` are described in
{doc}`data_quality`.

## Go deeper

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} The workflow, step by step
:link: tutorials/part_b/10_dynamic_workflow_glass_box
:link-type: doc

Open every step of a run and read its decisions.
:::

:::{grid-item-card} Interpretation settings
:link: api/settings
:link-type: doc

Every setting, its default and what it changes.
:::

:::{grid-item-card} Reservoir quality and net pay
:link: tutorials/part_a/09_reservoir_quality_and_net_pay
:link-type: doc

RQI, FZI, cutoffs and pay flags on real data.
:::
::::
