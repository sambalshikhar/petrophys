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

The trace covers the whole well. {meth}`~petrophys.workflow.DynamicWorkflowResult.group_records`
splits it as a context package does, each formation into groups of one
lithology, and gives every group its own record: the saturation routing and
its reasons, Archie coefficients per lithology, Rw at temperature, Rcl, the
SXO route and the cutoffs of its lithology family. These are the notes of the
group's context files:

```python
for group in result.group_records():
    print(group["formation"], group["group"], group["top"], group["bottom"])
    print(group["notes"]["saturation"])
```

`documents=True` adds the group's six context documents in full.

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

Net-pay cutoffs left unset take the standard of each sample's lithology family: clastic VCL < 0.40, PHIE > 0.08 and SW < 0.60; carbonate VCL < 0.30, PHIE > 0.045 and SW < 0.70.
Evaporite, coal, halite recognized on the logs and bad-hole samples are never net reservoir or net pay.
{doc}`data_quality` shows where to read both in the trace, and why a whole-well total can include marginal pay in shallow sections: read net pay per formation or over the reservoir interval.

(calculation-profiles)=
## Calculation profiles

`calculation_profile` selects an interpretation configuration. `"current"`
is the default; `"geolink"` provides the GeoLink evaluation configuration.
Explicitly supplied settings take precedence over profile defaults.
`result.summary()["settings_used"]` records the settings you supplied and
the `calculation_profile`; values the profile itself applies, such as its
net-pay cutoffs, show as `None` there and are recorded in the step decisions
of the trace.

For benchmark context, select a dataset `context_profile`: `geolink`, `costa`
or `danish`. The profile supplies the preparation, interpretation, calibration
and grouping rules for that collection. See {doc}`benchmark_context` for the
inputs and commands, and {doc}`context` for general context generation.

For an MCP evaluation harness, select the calculation configuration with
`petrophys-mcp serve --calculation-profile geolink`; see {doc}`mcp`.
The library API reads its profile from the supplied settings, while the MCP
server can also read `PETROPHYS_CALCULATION_PROFILE`.

## Settings, header values and input curves

Every setting has a default, and a value you set is used as given. The order is: a setting you set, then a LAS header value, then the default (for the "geolink" profile, its own default).
This holds for every route: `pp.interpret(well, **settings)`, `pp.interpret(well, config)`, `run_dynamic_workflow`, the batch processor, the zoned runner and `pp.export_context(workflow_config=...)`.
A setting you pass counts as set even when it equals the default (`rho_cl=2.45` keeps 2.45 against a header `RBSH`), also on a copy made with `dataclasses.replace`.
Archie `a`, `m`, `n` and the net-pay cutoffs left unset follow each sample's lithology (Archie) or lithology family (cutoffs). A single value, yours or the LAS header's (`M`, `SW_CUTOFF`, ...), holds for every sample. Set one per lithology and each sample takes its own; a lithology you leave out takes the header's value if there is one (`M_Z1` inside its zone), else its default, and `"*"` gives your value for every lithology not listed:

```python
result = pp.interpret(well, m={"sandstone": 1.8, "carbonate": 2.3},
                      sw_cutoff={"clastic": 0.5, "carbonate": 0.65})
```

Archie keys are lithologies of the Archie table (`sandstone`, `limestone`, `dolomite`, `shale`, `silt`, aliases such as `sand`) or a family (`clastic`, `carbonate`) for every lithology in it; cutoff keys are families, a lithology name standing for its family. The trace lists the values each lithology got (`archie_coefficients` in the saturation step, `cutoffs_by_family` in the net-pay step).

Settings can differ by depth: pass `zones`, a list of intervals each with its settings, or a table or CSV/Excel file with `top`, `bottom` and one column per setting (an empty cell sets nothing; `name` and `depth_unit` are optional):

```python
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, zones=[
    {"name": "overburden", "top": 176, "bottom": 1908, "rw": 0.5, "rw_meas_temp_f": 60},
    {"name": "Brent", "top": 2841, "bottom": 2958, "m": 1.9},
])
```

Each zone's settings are run over the whole well and its depths merged in, so a zone changes what its settings change and nothing else: the picks taken from the data (GR baselines, irreducible water saturation, the core model) stay those of the whole well. `result.zone_results` holds one run per distinct set of zone settings, and `result.warnings` their messages. Zones may not overlap; whole-run settings such as `calculation_profile`, `skip_steps` and the lithology inputs cannot differ by zone; a zone outside the logged depths is noted and not used.
The full order is: your zone values inside their zone, then your whole-well settings, then LAS header zone values (`RW_Z1`, `M_Z1`, ... with `TOP_Z1`/`BOT_Z1`) inside their zone, then LAS header whole-well values, then the defaults. A header value one of yours overrides is named in the warnings. The same order holds on every route; `DynamicWorkflow(config).run` alone uses the config as given, without the header.
An explicit setting wins over a LAS header value: `rw` over the header RW, `salinity_ppm` over the header RW, `rw_meas_temp_f` over the header RWT, and `rm`, `rmf_salinity_ppm` and `rmf_chlorides_mg_l` over the header RMF and CLFL (a value `pp.interpret` copied from the header, such as CLFL, ranks with the header, below a direct RMF).
`phi_cl`, `rmf_meas_temp_f` and `rm_meas_temp_f` are unset (`None`) by default and resolve to 0.15 v/v and 75 degF, so an explicit value, including one equal to the default, is always used.
Zone-only header parameters (`RW_Z1` with `TOP_Z1`/`BOT_Z1`) apply inside their zone; outside it the whole-well settings and header values are used, and the trace and warnings say so.
A zone whose `TOP_Z1`/`BOT_Z1` span every depth of the well replaces the header value everywhere, in `pp.interpret` as in the zoned runner, and a warning names both values when they differ.
Header values are converted from their units: `TD` from its own depth unit (the well's depth unit when it has none), `RHO_MA`/`RHO_FL` from kg/m3 (`KG/M3`, `K/M3`) or lb/ft3, and salinity from ppm, ppk, ppt (g/kg), mg/l, g/l or %; a salinity in a unit not on that list is not used, with a warning.
Settings outside their physical range are refused: temperatures below absolute zero, `bht_f` below `surface_temp_f`, `rho_ma` outside 1.5-3.5 g/cm3 or not above `rho_fl`, `rho_fl` outside 0-2 g/cm3, `nphi_ma` outside -0.1 to 0.5 and `salinity_ppm` outside 0-400,000 ppm.
`rw_meas_temp_f` given without `rw` is the measurement temperature of the header RW, replacing its RWT.
A measured `TEMP` log with at least half of its samples finite is the temperature source, with the modelled gradient filling its gaps; a sparser log is kept as `TEMP_LOG` and the modelled profile is used. With `bht_f` or `surface_temp_f` set, the modelled profile is used and the log is kept as `TEMP_LOG`.
Set `rho_ma`, `dt_ma` or `nphi_ma` and it holds at every depth, also where a lithology log would give each group its own matrix value.
The SP-derived Rw uses `sp_max` and `sp_min` as its clean-sand and shale lines when they are set.
With `apply_hc_correction=True`, the porosity step reads the hydrocarbon-corrected `NPHI_HC` and `RHOB_HC`. The correction scales with the flushed zone's hydrocarbon (1 - SXO) and runs before the saturation step, so it needs an input SXO curve; without one it is skipped with a warning.
`PHIE_CC` is written by the core-correction step alone, before saturation, so SW, BVW, net pay and RQI read the same porosity; the step runs on core porosity named `PHIC`, `CPOR`, `CPOR-AMB` or `CPOR_AMB`, or on peer-well core (`use_peer_core=True` with `peer_wells`).
A peer well needs PHIE or PHIT (interpret it first; one without is left out, with a warning), and its formations may be attached with `pp.read(..., formations=...)`.
When the run fits its own core model, input `PHIE_CC` and `PERM_CC` curves are kept as `PHIE_CC_INPUT` and `PERM_CC_INPUT` and not used; without core data, or with `core_correction` in `skip_steps`, input `PHIE_CC` and `PERM_CC` are used as they stand by every step, and a warning says so.
`PERM_CC` is written only where the core-corrected porosity lies within the porosity range of the plugs behind the poro-perm fit (the samples set aside are counted in the trace).
`lithology_domain="auto"` without a lithology log or core matrix evidence is settled before the core-correction step, from the mineral solver's volumes or else the clastic default, so Rw, SW and net pay use the domain that `result.config` reports.
Without a clay indicator (gamma ray, SP or neutron-density) net pay assumes VCL = 0, and a warning says so.
The `"geolink"` profile keeps the GeoLink eval package's calculation in each case.

## Clay volume

`pp.interpret` takes VCL as the minimum of the gamma-ray and neutron-density indicators, VCL_GR and VCL_ND.
VCL_ND is measured from the neutron-density clean line of the run's matrix, to a clay point picked in the well's good-hole shales: the median RHOB and NPHI of 20 or more shale-lithology samples, else of the samples with VCL_GR above 0.9, else (2.62 g/cm3, 0.38).
One median over the whole log is lighter than the shales of a deep reservoir where shallow, undercompacted shale dominates, so set `nd_clay` from a crossplot of the interval you evaluate.
A clay point less than 0.10 v/v of neutron above the clean line leaves VCL_ND out of VCL, with a warning.

VCL_SP is reported, and joins the minimum only when you set both SP baselines: `sp_min`, the shale baseline, and `sp_max`, the clean-sand line.
Both are constant lines over the whole log, so set them only for an SP without baseline drift, such as a drift-corrected SP.
Equal baselines are rejected, and a warning flags baselines whose order contradicts the SP polarity (read from the SP and GR, and taken as normal where they do not show it).

## Water saturation

With `sw_model="auto"` (the default), saturation is routed per sample.
Samples with VCL below `sw_routing_vcl_cutoff` (by default `vcl_cutoff`, or 0.3 when that is unset) get Archie; the same clay volume bounds the samples SWIRR_APP is picked from.
With `vcl_cutoff` unset, routing switches at 0.3 while clastic net rock extends to VCL 0.40, so net rock with 0.3-0.4 clay gets a shaly-sand equation.
In shaly rock, carbonate samples get Dual Water when the bound-water resistivity `rwb` is given; Rwb is never estimated.
All other shaly samples get Waxman-Smits when `cec` (meq/100 g) and `b_coeff` ((S/m)/(meq/cm3)) are given, Simandoux when `salinity_ppm` is at least 20,000 ppm, and otherwise Indonesia, the published equation of Poupon and Leveaux (1971).
`sw_model="dual_water"` without `rwb`, or `"waxman_smits"` without `cec` and `b_coeff`, fails the saturation step with a message naming the missing input.
With oil-based mud (`mud_type="OBM"`), no SXO or MHI is computed and Rw is not estimated from the SP or Rxo/Rt.

## Lithology domain

`lithology_domain` picks the mineral solver and keeps the automatic matrix within its family. It defaults to `clastic`: set `carbonate` for carbonate wells. It is not a per-depth rock label: lithology intervals from {doc}`input_templates` supply those, else a lithology code curve in the LAS, and both take precedence over computed labels.

| Domain | Mineral solver | Needs |
|---|---|---|
| `clastic` (default) | multi-mineral solve ({class}`~petrophys.interpretation.lithology.MultiMineralSolverV1`): each tool equation weighted by its nominal uncertainty, and samples whose mineral total misses the available rock volume by more than 0.05 left unsolved; quartz, calcite, dolomite and anhydrite with PEF (sonic beside it), quartz, calcite and dolomite from RHOB and NPHI without it; PEF set aside when it reads like barite-weighted mud; samples with PEF but no NPHI solved by {class}`~petrophys.interpretation.lithology.MultiMineralSolver` (version 2, porosity solved jointly); sandstone matrix | RHOB, NPHI or PEF, DTC, VCL, PHIE |
| `carbonate` | calcite and dolomite from bulk density and closure, with clay from VCL at `rho_cl` and pore fluid `rho_fl`; `carbonate_calibration="costa"` applies the COSTA field calibration instead (illite from GR at 2.52 g/cm3 and a 1.1 g/cm3 brine, unless `rho_cl` or `rho_fl` is set); limestone or dolomite matrix | RHOB, VCL (GR for `costa`), PHIE or PHIT |
| `evaporite` | the clastic mineral solve (anhydrite among its minerals); an automatic matrix with no family limit, taken from the lithology log or core where they exist, halite endpoints where the logs read like halite; clastic cutoffs, with evaporite, halite and anhydrite samples left out of net reservoir and pay. On a well whose matrix resolves to sandstone it gives the clastic result | as `clastic` |
| `auto` | resolved from attached or embedded geology; falls back to clastic with a warning | |

If the data point to a carbonate while the domain is `clastic`, the run keeps the sandstone matrix and warns you to set `lithology_domain="carbonate"`.

The lithology code curve is the curve `lithology_curve` names, else `LITHOLOGY_GEOLINK`, else the first curve under a lithology-code name (`LITHOLOGY`, `LITHOFACIES`, `FACIES`, `LITH_CODE`, `ROCK_TYPE`, ...) whose values pass the code checks in {doc}`input_templates`.
That one curve gives the vendor lithology, the `"auto"` matrix and the per-depth lithology groups, and its codes are decoded with the built-in lithology code table (Kingdom nomenclature) whatever the curve is called.
`result.warnings` names a curve other than `LITHOLOGY_GEOLINK`; if it holds other codes, pass its codebook as `lithology_codes`, or `lithology_source="computed"`, or lithology intervals.

The carbonate solve uses bulk density alone, so its calcite-dolomite split depends on the pore-fluid density `rho_fl` (1.0 g/cm3 by default, 1.1 with `carbonate_calibration="costa"`; a value you set is used by porosity and the mineral solve alike) and on the matrix density behind the porosity.
On HW-3 the generic defaults put nearly nine samples in ten at the calcite end member, with no dolomite; set `rho_fl` for your brine, and use the field calibration on COSTA wells.
Here the clastic run on 30/6-5 is set against a carbonate run on the COSTA well HW-3:

```{code-cell} python
# Rw comes from the LAS header; COSTA wells take the field's own calibration.
hw3 = pp.interpret(pp.read(pp.sample("HW-3")), lithology_domain="carbonate",
                   carbonate_calibration="costa")

def lithology_step(run):
    return next(record for record in run.steps_executed if record.step == "lithology")

rows = []
for name, run in {"30/6-5": result, "HW-3": hw3}.items():
    step = lithology_step(run)
    rows.append({"well": name, "domain": run.lithology_domain, "method": step.method,
                 "solved": f"{step.parameters['multi_mineral_solved_samples']} of "
                           f"{step.parameters['multi_mineral_total_samples']}",
                 "outputs": ", ".join(step.output_curves)})
print("HW-3 illite:", lithology_step(hw3).parameters["clay_source"])
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

The carbonate solve is closed-form over whole arrays, through the apparent matrix density (Clavier and Rust, 1976), so a whole well takes well under a second; the "geolink" calculation profile keeps the COSTA notebooks' sample-by-sample solve, which takes about a second on a COSTA interval but minutes on a whole well.
Where the solved volumes miss RHOB by more than 0.02 g/cm3 (an apparent matrix density outside calcite-dolomite, such as quartz, anhydrite or bad hole), mineral volumes are NaN and QC-flagged; closure alone does not prove a mineral model right.

## Zoned and batch runs

{func}`~petrophys.workflow.process_las_directory_with_workflow` interprets every LAS file in a folder and writes the interpreted LAS plus a JSON trace per well.
Each file runs through {func}`~petrophys.workflow.run_dynamic_workflow_with_zones`, where LAS header values fill settings you leave unset. Header zones such as `TOP_Z1`, `BOT_Z1`, `RW_Z1` re-run their depth range with their own settings, from the input curves rather than the whole-well run's results; a curve the zone run did not compute (for example `PHIE_CC` where the zone holds too few core plugs) is NaN in the zone after the merge, and a warning names it.

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
        config=pp.DynamicWorkflowConfig(lithology_domain="carbonate",
                                        carbonate_calibration="costa"))
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
