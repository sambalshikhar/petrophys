---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Terms and curves

A reference for the names in results, traces and settings. The curve table is
generated from a real run on 30/6-5, so it always matches the installed
version. The workflow steps are listed in {ref}`workflows-steps`.

```{code-cell} python
:tags: [remove-cell]
import warnings
import pandas as pd

# Display settings for this page: whole table cells, warnings without file paths.
pd.set_option("display.max_colwidth", None)
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
# The bundled LAS stores its SWC/SOC core saturations as percentages under a 'frac'
# label; Petrophys converts them and warns. Hide only that message.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

## Terms

| Term | Meaning |
| --- | --- |
| Lithology domain | The mineral model for the whole run: `clastic`, `carbonate`, `evaporite` or `auto` (`lithology_domain`). Not a per-depth label. |
| Lithology intervals | Optional per-depth rock labels from a table. They override labels in the LAS and from the mineral solve. |
| Lithology code curve | A per-sample curve of lithology codes in the LAS: the curve named by `lithology_curve`, else `LITHOLOGY_GEOLINK`, else a curve under another lithology-code name such as `LITHOLOGY` or `FACIES`. Its codes are decoded with the built-in lithology code table (Kingdom nomenclature), or with a user codebook passed as `lithology_codes`; see {doc}`input_templates`. |
| Formations | Optional named depth intervals (groups, formations, members) for summaries, plots and core calibration. |
| Rw | Formation-water resistivity in ohm.m, at a stated temperature (`rw_meas_temp_f`). The workflow corrects it to formation temperature. |
| Cutoffs | Limits on clay volume, porosity and water saturation (`vcl_cutoff`, `phie_cutoff`, `sw_cutoff`). Left unset, each sample takes the standard of its lithology family, clastic or carbonate. |
| Net rock, reservoir, pay | Samples passing the clay cutoff; then also the porosity cutoff; then also the saturation cutoff. Evaporite, coal, halite recognized on the logs and bad-hole samples are never net reservoir or pay. |
| Bad hole | `BAD_HOLE` = 1: the caliper reads more than 1.5 in over the bit size, the bit is larger than 17.5 in, or the absolute density correction DRHO is above 0.15 g/cm3. The caliper and bit-size tests run only where a bit size is known; see {ref}`review-inputs`. |
| Class tables | The classes each property is graded into, such as `CLEAN_FORMATION` for clay volume or `GOOD_RESERVOIR` for reservoir quality, with a `NO_*_DATA` label where a value is missing: `CLASS_TABLES` in `petrophys.model`. Context labels and the diagnostics analyzers use them; see {doc}`context`. |
| Calculation profile | An interpretation configuration (`calculation_profile`): `"current"` is the default and `"geolink"` supplies the GeoLink evaluation configuration; see {ref}`calculation profiles <calculation-profiles>`. |
| Trace | The record of a run: each step's status, method, inputs, outputs, parameters and decisions. |
| Context | A compact JSON summary of an interpreted well for LLMs and other tools; see {doc}`context`. |

## Where each input goes

Formations, lithology intervals and lithology codebooks go in as `formations=`,
`lithology=` and `lithology_codes=` (on the command line `--formations`,
`--lithology` and `--lithology-codes`); their layouts and readers are in
{doc}`input_templates`. `pp.read` takes formations and lithology intervals
only. In `pp.interpret`, `lithology=` also takes the name of the lithology code
curve (`"FACIES"`) or a lithology domain (`"carbonate"`). Through MCP,
`load_well` takes `formations` and `lithology`, and `interpret_well` takes the
settings, a codebook among them as the `lithology_codes` setting.

:::{warning}
In `pp.read`, `lithology=` attaches lithology intervals only. As a
`DynamicWorkflowConfig` setting, `lithology` is an alias of
`lithology_domain`; use `lithology_domain` there.
:::

## Curves the workflow adds

Every curve `pp.interpret` added on 30/6-5, with the unit and description it
carries and the fraction of depths it covers. Other inputs can add or omit
curves; fractions are in v/v. The trace, `result.to_dataframe()`, names the step
that wrote each one. The clay volumes are kept under a second name for other readers, with the same values: `VCLAY` for `VCL`, `VCLGR` for `VCL_GR`, `VCLSP` for `VCL_SP` and `VCLND` for `VCL_ND`.

```{code-cell} python
import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
curves = pd.DataFrame(result.well.describe()["curves"])
curves[~curves["mnemonic"].isin(well.curves)]
```

The cutoffs behind `NET_ROCK`, `NET_RES` and `NET_PAY` are given under
{ref}`workflows-settings`, and {ref}`review-cutoffs` shows how to read them in
the trace.

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
