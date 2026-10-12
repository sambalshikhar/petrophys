---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Quickstart

Read a bundled well, plot it, interpret it and export the result. Every output
on this page is produced when the documentation is built.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import textwrap
import warnings
import pandas as pd

# Display settings for this page: whole table cells, warnings wrapped and without file paths.
pd.set_option("display.max_colwidth", None)
warnings.formatwarning = lambda message, category, *args, **kwargs: textwrap.fill(
    f"{category.__name__}: {message}", 88, subsequent_indent="    ") + "\n"
```

```{code-cell} python
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))   # or the path to your own LAS file
print(well)
print(f"{well.top:.1f} - {well.bottom:.1f} {well.depth_unit}")
```

## Plot the raw logs

`pp.plot` draws the curves already in the file, so no interpretation is needed.
The window below covers the Brent reservoir, at 2841–2958 m.

```{code-cell} python
viewer = pp.plot(
    well,
    depth_range=(2800, 3000),
    tracks=["LITHOLOGY", "GAMMA / HOLE", "RESISTIVITY", "DENSITY / NEUTRON"],
    figsize=(9.5, 10),
)
viewer.show()
```

## Interpret

`pp.interpret` works on a copy of the well. Settings are keywords: here the
formation-water resistivity `rw` and the temperature it was measured at.
{doc}`api/settings` lists them all.

```{code-cell} python
from textwrap import fill

result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)   # Rw 0.07 ohm.m at 230 degF

print("status:", result.status)
print("resolved:", result.config.matrix_type, "matrix,",
      result.config.lithology_domain, "domain")
for message in result.warnings:
    print(fill(f"warning: {message}", 88, subsequent_indent="    "))
```

:::{note}
These warnings are real: this LAS labels its core saturations `SWC` and `SOC`
as fractions, but they hold percentages. Petrophys divides them by 100 for its
core calibration and says so each time it reads them, so each message appears
twice. They are Python warnings, not entries of `result.warnings`, which is
empty for this run. The rest of this page hides only these warnings; see
{ref}`core-saturation-units`.
:::

```{code-cell} python
:tags: [hide-input]
import warnings

# Hide only the SWC/SOC unit message explained above; other warnings stay visible.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

| `result.status` | Meaning |
| --- | --- |
| `"interpreted"` | Steps ran and produced interpretation curves. |
| `"no_interpretation"` | Nothing failed, but the inputs supported no interpretation step. |
| `"failed"` | A step failed; `result.errors` says why and `result.raise_for_errors()` raises. |

A step that cannot run is skipped with a reason, never filled with invented
defaults:

```{code-cell} python
import pandas as pd

skipped = pd.DataFrame(result.specific_steps_skipped.items(),
                       columns=["Step", "Why it did not run"])
skipped.style.hide(axis="index")
```

:::{admonition} Where Rw comes from
:class: note

Without `rw`, the workflow uses the first source it finds:

1. an `RW` value in the LAS header;
2. a formation-water salinity, `salinity_ppm`;
3. the SP or Rxo/Rt response, given a mud-filtrate resistivity (water-based mud
   only);
4. the apparent Rw of clean, wet rock, with a warning;
5. a default of 0.05 ohm.m, with a warning (`allow_default_rw=False` turns it
   off).

Always give the temperature Rw was measured at (`rw_meas_temp_f`); otherwise Rw
is taken at the well's median formation temperature, with a warning.
The 30/6-5 header gives `RW = 0.7` with no temperature, so this page passes its
own value.
:::

The `rw` step of the trace shows which source was used:

```{code-cell} python
rw_step = next(step for step in result.steps_executed if step.step == "rw")
print(rw_step.method)
print(rw_step.decisions)
```

## Summarize the Brent reservoir

The log starts in shallow, poorly consolidated rock that can pass the pay
cutoffs (the clay, porosity and saturation limits). One Rw also serves the whole
well, while shallow formation waters are usually fresher than the reservoir's,
so shallow sections that no lithology code covers can be flagged as marginal
pay; {doc}`data_quality` explains this limitation. Summarize the interval you
are evaluating, not the whole well.

```{code-cell} python
print("whole well:", result.summary()["net_thickness"])
brent = result.summary(top=2841, bottom=2958)
print("Brent:     ", brent["net_thickness"])

stats = pd.DataFrame(brent["key_results"]).T
stats[["p10", "median", "p90"]].astype(float).round(3).assign(unit=stats["unit"])
```

## Read the trace

Every executed step records its method, inputs, outputs, parameters and
decisions. `result.print_trace()` prints all of them; the table below keeps
three columns.

```{code-cell} python
from textwrap import shorten

trace = result.to_dataframe()
trace["Outputs"] = trace["Output Curves"].map(lambda text: shorten(text, 40))
trace[["Step", "Status", "Method", "Outputs"]].style.hide(axis="index")
```

## Plot the interpretation

Passing the result plots the interpreted well. The `petrophysics` template adds
porosity, saturation, permeability and pay-flag tracks.

```{code-cell} python
pp.plot(
    result,
    depth_range=(2841, 2958),
    template="petrophysics",
    tracks=["LITHOLOGY", "GR / VCL", "RESISTIVITY", "POROSITY",
            "SATURATION", "PERMEABILITY", "FLAGS"],
    formation_name="Brent reservoir",
    figsize=(12, 10),
).show()
```

## Export

`to_las` writes every curve plus the settings used (`PP_*` parameters).
`to_json` writes the complete trace.

```{code-cell} python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as folder:
    las_path = result.to_las(Path(folder) / "30_6-5_interpreted.las")
    json_path = result.to_json(Path(folder) / "30_6-5_trace.json")
    for path in (las_path, json_path):
        print(f"{path.name}: {path.stat().st_size / 1e3:,.0f} kB")
    for line in las_path.read_text().splitlines():
        if line.startswith("PP_"):          # the settings used, in the ~Params section
            name, *_, value = line.split(":")[0].split()
            if value.replace(".", "", 1).isdigit():
                value = f"{float(value):.3g}"
            print(f"  {name:<11} {value}")
```

## Next steps

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Your own data
:link: own_data
:link-type: doc

Read CSV logs, attach formation and lithology tables, and check depths.
:::

:::{grid-item-card} Interpretation workflows
:link: workflows
:link-type: doc

Choose methods, override settings, run zones and batches.
:::

:::{grid-item-card} Complete tour
:link: tutorials/complete_tour
:link-type: doc

Every capability of the library on bundled data, in one notebook.
:::

:::{grid-item-card} Case study: 30/6-5
:link: tutorials/case_studies/cs1_full_formation_evaluation_30_6-5
:link-type: doc

A full formation evaluation of this well, step by step.
:::
::::
