---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Workflows

{func}`pp.interpret <petrophys.api.interpret>` runs a complete interpretation of one well.
It runs every step the available curves and settings allow, and records why the others were skipped.

```{code-cell} python
:tags: [hide-cell]
%matplotlib inline
import warnings

import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full

# 30/6-5 stores core saturation in percent under a "frac" label. Petrophys
# converts it and warns; see "Reviewing an interpretation". Only that message is hidden.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## Interpret a well

```{code-cell} python
import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)  # Rw 0.07 ohm.m at 230 degF
print(result.status, "|", result.lithology_domain, "|",
      len(result.steps_executed), "trace records")
```

```{code-cell} python
:tags: [remove-input]
import matplotlib.pyplot as plt
from petrophys.visualization import use_plot_style
use_plot_style()
from matplotlib.patches import FancyBboxPatch, Patch
from petrophys.workflow import workflow_steps

TEAL, GREY = "#087f73", "#8a949c"
steps = [step["name"] for step in workflow_steps()]
ran = {record.step for record in result.steps_executed}
COLUMNS = 6
fig, ax = plt.subplots(figsize=(8, 4.5))
for index, name in enumerate(steps):
    row, col = divmod(index, COLUMNS)
    done = name in ran
    ax.add_patch(FancyBboxPatch(
        (col + 0.05, -row + 0.08), 0.9, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=TEAL if done else "white", edgecolor=TEAL if done else GREY,
        hatch=None if done else "////", linewidth=1.2))
    backing = None if done else dict(facecolor="white", edgecolor="none", pad=1.5)
    ax.text(col + 0.5, -row + 0.7, str(index + 1), ha="center", va="center", fontsize=7.5,
            color="white" if done else "#3b4550", bbox=backing)
    label = name.replace("_", "\n", 1)
    ax.text(col + 0.5, -row + 0.4, label, ha="center", va="center",
            fontsize=8.5 if max(map(len, label.split("\n"))) <= 10 else 7.5,
            color="white" if done else "#3b4550",
            fontweight="bold" if done else "normal", bbox=backing)
ax.set_xlim(0, COLUMNS)
ax.set_ylim(-3.05, 1.0)
ax.axis("off")
executed = sum(name in ran for name in steps)
ax.set_title(f"30/6-5: {executed} of {len(steps)} pipeline steps ran, "
             f"{len(steps) - executed} were skipped (in pipeline order)", fontsize=11)
ax.legend(handles=[Patch(facecolor=TEAL, label="ran"),
                   Patch(facecolor="white", edgecolor=GREY, hatch="////", label="skipped")],
          loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
plt.show()
```

## Read the trace

Each step records its method, inputs, outputs, parameters and decisions.
{meth}`~petrophys.workflow.DynamicWorkflowResult.to_dataframe` gives one row per step; here are six key steps with their first two decisions.

```{code-cell} python
trace = result.to_dataframe()
key = trace["Step"].isin(["temperature", "vclay", "lithology", "rw",
                          "saturation", "net_pay"])
key_steps = trace.loc[key, ["Step", "Method", "Decisions"]]
first_two = key_steps["Decisions"].str.split(" | ", regex=False).str[:2]
key_steps["Decisions"] = first_two.str.join("; ")
key_steps.style.hide(axis="index")
```

Skipped steps keep their reason:

```{code-cell} python
skipped = pd.DataFrame(result.specific_steps_skipped.items(), columns=["Step", "Reason"])
skipped.style.hide(axis="index")
```

{meth}`~petrophys.workflow.DynamicWorkflowResult.print_trace` prints the full record, and {meth}`~petrophys.workflow.DynamicWorkflowResult.to_json` saves it.

## The steps

The pipeline runs in this order. Use these names in `skip_steps`.

```{code-cell} python
from petrophys.workflow import workflow_steps

pd.DataFrame(workflow_steps()).style.hide(axis="index")
```

## Settings

Settings are {class}`~petrophys.workflow.DynamicWorkflowConfig` fields, passed as keywords or as one config object.
Invalid values fail before anything runs, with the accepted values:

```{code-cell} python
from textwrap import fill

try:
    pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model="archy", hc_type="gass")
except ValueError as error:
    for line in str(error).splitlines():
        print(fill(line, 88, subsequent_indent="      "))
```

The most-used settings, from {func}`~petrophys.workflow.config_schema`. {doc}`api/settings` lists them all.

```{code-cell} python
from petrophys.workflow import config_schema

used_most = ["rw", "rw_meas_temp_f", "bht_f", "lithology_domain", "matrix_type", "gr_min",
             "gr_max", "sw_model", "vcl_cutoff", "phie_cutoff", "sw_cutoff", "skip_steps"]
schema = {item["name"]: item for item in config_schema()}
pd.DataFrame([{"setting": name, "default": repr(schema[name]["default"]),
               "description": schema[name]["description"]}
              for name in used_most]).style.hide(axis="index")
```

In the well-parameter template, `gr_clean` and `gr_shale` map to `gr_min` and `gr_max`.

## Lithology domain

`lithology_domain` picks the mineral solver. It is not a per-depth rock label: lithology intervals from {doc}`input_templates` supply those, and take precedence over computed labels.

| Domain | Mineral solver | Needs |
|---|---|---|
| `clastic` | canonical multi-mineral solve with a closure gate | RHOB, NPHI, VCL (PEF or DTC add information) |
| `carbonate` | COSTA method: illite from GR, then a bounded calcite-dolomite solve | RHOB, GR, PHIT or PHIE |
| `auto` | resolved from attached or embedded geology; falls back to clastic with a warning | |

Here the clastic run on 30/6-5 is set against a carbonate run on the COSTA well HW-3:

```{code-cell} python
# Rw comes from the LAS header.
hw3 = pp.interpret(pp.read(pp.sample("HW-3")), lithology_domain="carbonate")

def lithology_step(run):
    return next(record for record in run.steps_executed if record.step == "lithology")

rows = []
for name, run in {"30/6-5": result, "HW-3": hw3}.items():
    step = lithology_step(run)
    rows.append({"well": name, "domain": run.lithology_domain, "method": step.method,
                 "solved": f"{step.parameters['multi_mineral_solved_samples']} of "
                           f"{step.parameters['multi_mineral_total_samples']}",
                 "outputs": ", ".join(step.output_curves)})
print("HW-3 illite:", lithology_step(hw3).parameters["illite_equation"])
pd.DataFrame(rows).style.hide(axis="index")
```

```{code-cell} python
:tags: [remove-input]
logs = hw3.well.to_dataframe()
depth = logs["DEPTH"]
fig, (ax_gr, ax_rho, ax_min) = plt.subplots(
    1, 3, figsize=(8, 4.5), sharey=True, gridspec_kw={"width_ratios": [1, 1, 2]})
ax_gr.plot(logs["GR"], depth, color="#3b4550", lw=1)
ax_gr.set_xlabel("GR (API)")
ax_rho.plot(logs["RHOB"], depth, color="#e34948", lw=1)
ax_rho.set_xlabel("RHOB (g/cm3)")
left = 0
for curve, label, color in [("VILL", "illite", "#008300"), ("VCALC", "calcite", "#2a78d6"),
                            ("VDOLO", "dolomite", "#e87ba4"), ("PHIT", "porosity", "#d9f2ea")]:
    ax_min.fill_betweenx(depth, left, left + logs[curve], color=color, lw=0, label=label)
    left = left + logs[curve]
ax_min.set_xlim(0, 1)
ax_min.set_xlabel("volume fraction")
ax_min.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax_gr.set_ylabel(f"depth ({hw3.well.depth_unit})")
ax_gr.invert_yaxis()
for axis in (ax_gr, ax_rho, ax_min):
    axis.grid(alpha=0.3)
fig.suptitle("HW-3, Kharaib-2: carbonate mineral volumes from GR, RHOB and porosity", fontsize=11)
fig.tight_layout()
plt.show()
```

The carbonate solve runs sample by sample: about 1 s on a 200-sample COSTA interval, but about 10 minutes on 20,000 samples. Run it on carbonate intervals, not whole wells.
Where the logs cannot constrain a solve, mineral volumes are NaN and QC-flagged; closure alone does not prove a mineral model right.

## Zoned and batch runs

{func}`~petrophys.workflow.process_las_directory_with_workflow` interprets every LAS file in a folder and writes the interpreted LAS plus a JSON trace per well.
Each file runs through {func}`~petrophys.workflow.run_dynamic_workflow_with_zones`, where LAS header values fill settings you leave unset. Header zones such as `TOP_Z1`, `BOT_Z1`, `RW_Z1` re-run their depth range with their own settings.

```{code-cell} python
import shutil
import tempfile
from pathlib import Path

from petrophys.workflow import process_las_directory_with_workflow

with tempfile.TemporaryDirectory() as folder:
    incoming = Path(folder, "incoming")
    incoming.mkdir()
    for name in ["HW-3", "HW-4", "HW-5"]:
        shutil.copy(pp.sample(name), incoming)
    batch = process_las_directory_with_workflow(
        incoming, Path(folder, "interpreted"),
        config=pp.DynamicWorkflowConfig(lithology_domain="carbonate"))
    written = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*.*"))

print("\n".join(written))
pd.DataFrame([{
    "well": Path(item.source_file).stem,
    "success": item.success,
    "zones": item.zones_applied,
    "new curves": len(item.computed_curves),
    "net pay (ft)": item.result.base_result.summary()["net_thickness"]["net_pay"],
    "first warning": item.warnings[0] if item.warnings else "",
} for item in batch]).style.hide(axis="index").format(precision=1)
```

Failed files are reported, not raised. {func}`~petrophys.workflow.write_batch_report` and {func}`~petrophys.workflow.write_curve_audit_report` turn the results into CSV reports.

## Run selected steps

{meth}`DynamicWorkflow.run_selected_steps <petrophys.workflow.DynamicWorkflow.run_selected_steps>` runs a subset in pipeline order. Unlike `pp.interpret`, it adds curves to the well you pass.

```{code-cell} python
from petrophys.workflow import DynamicWorkflow, WorkflowStep

workflow = DynamicWorkflow(pp.DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230))
partial = workflow.run_selected_steps(pp.read(pp.sample("30_6-5")),
                                      [WorkflowStep.VCLAY, WorkflowStep.POROSITY])
for record in partial.steps_executed:
    outputs = ", ".join(record.output_curves[:5])
    print(f"{record.step:14s} {record.method:28s} {outputs}")
```

The older {class}`~petrophys.workflow.standard_workflow.StandardWorkflow` is deprecated and listed under Legacy in the {doc}`API reference <api/index>`.

## Go deeper

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} The workflow as a glass box
:link: tutorials/part_b/10_dynamic_workflow_glass_box
:link-type: doc
Every step, decision and skip reason on 30/6-5.
:::

:::{grid-item-card} Lithology and the mineral solver
:link: tutorials/part_a/05_lithology_and_mineral_solver
:link-type: doc
How the clastic solve works, and when it cannot.
:::

:::{grid-item-card} Carbonate rock typing
:link: tutorials/case_studies/cs2_carbonate_rock_typing_costa
:link-type: doc
A COSTA carbonate well from logs to rock types.
:::

:::{grid-item-card} Multi-well field study
:link: tutorials/part_b/18_multiwell_field_study
:link-type: doc
Run and compare many wells at once.
:::
::::
