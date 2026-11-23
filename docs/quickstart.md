---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
  execution_in_temp: true
---

# Quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambalshikhar/petrophys/blob/main/examples/quickstart.ipynb)

Run a first well: read a bundled well, plot it, interpret it and export the
result. Every output shown is computed from that well. Run it with Petrophys
installed ({doc}`installation`), or open it in Google Colab with the button
above and run it in your browser, with nothing to install
({doc}`new_to_python`).

```{code-cell} python
:tags: [remove-cell]
import textwrap, warnings
import pandas as pd

# Display settings for this page: whole table cells, three decimals, wrapped warnings.
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.3f}".format)
warnings.formatwarning = lambda message, category, *args, **kwargs: textwrap.fill(
    f"{category.__name__}: {message}", 88, subsequent_indent="    ") + "\n"
```

```{code-cell} python
import pandas as pd
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))   # or the path to your own LAS file
well
```

The well has 20 curves, logged from 177 m to 3556 m.

## Plot the raw logs

`pp.plot` plots a well or an interpretation. Here it plots the logs as read,
before any interpretation, and `tracks=` picks four of the layout's tracks. The
window, 2800–3000 m, brackets the Brent Group (2841–2958 m).

```{code-cell} python
pp.plot(well, top=2800, bottom=3000,
        tracks=["LITHOLOGY", "GAMMA / HOLE", "RESISTIVITY", "DENSITY / NEUTRON"])
```

## Interpret

`pp.interpret` works on a copy of the well. Settings are keywords: here the
formation-water resistivity `rw` and the temperature it was measured at.
{doc}`api/settings` lists them all.

```{code-cell} python
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)   # Rw 0.07 ohm.m at 230 degF
result.status
```

`result.warnings` holds what the run wants you to check:

```{code-cell} python
result.warnings
```

The first says the photoelectric log (PEF) reads like barite-weighted mud, so
the minerals were solved without it. The second says how much net pay the
bad-hole flag removed over the whole well; {ref}`review-inputs` explains that
flag.

:::{note}
The two `UserWarning` messages above the raw-log plot are real: this LAS labels
its core saturations `SWC` and `SOC` as fractions, but they hold percentages.
Petrophys divides them by 100 for plotting and core calibration and says so when
it reads them (Python shows a warning once per place it is raised, so the
interpretation does not repeat them here). They are Python warnings, not entries
of `result.warnings`. The rest of this page hides only these warnings; see
{ref}`core-saturation-units`.
:::

```{code-cell} python
:tags: [remove-cell]
# Hide only the SWC/SOC unit message explained above; other warnings stay visible.
warnings.filterwarnings("ignore", message=r"(SWC|SOC): inferred percent core saturation")
```

| `result.status` | Meaning |
| --- | --- |
| `"interpreted"` | Steps ran and produced interpretation curves. |
| `"no_interpretation"` | Nothing failed, but the inputs supported no interpretation step. |
| `"failed"` | A step failed; `result.errors` says why and `result.raise_for_errors()` raises. |

To see the original exception while debugging, pass `raise_on_error=True`.

A step that cannot run is skipped with a reason, never filled with invented
defaults:

```{code-cell} python
result.specific_steps_skipped
```

:::{admonition} Where Rw comes from
:class: note

The workflow takes Rw from the first source it finds, your settings before the
file's header:

1. the Rw you set, `rw` (at `rw_meas_temp_f`);
2. the formation-water salinity you set, `salinity_ppm`;
3. the `RW` value in the LAS header (at its `RWT`, or at `rw_meas_temp_f` when
   you set one, else at the header formation temperature `FMT`);
4. a formation-water salinity in the LAS header (`WATSAL`, `WSAL`, `SAL`, ...);
5. the SP or Rxo/Rt response, given a mud-filtrate resistivity (water-based mud
   only);
6. the apparent Rw of clean rock, with a warning: each sample's apparent Rw
   is corrected from its own temperature to 75 degF, and the P5 of those
   values is taken as one water for the whole well (values saltier than
   saturated brine are left out). Use `zones=` where the water changes, for
   example fresher shallow sands;
7. a default of 0.05 ohm.m, with a warning (`allow_default_rw=False` turns it
   off).

Always give the temperature Rw was measured at (`rw_meas_temp_f`); otherwise Rw
is taken at the well's median formation temperature, with a warning.
The 30/6-5 header gives `RW = 0.07` ohm.m at `RWT = 230` degF; this page
passes the same values as settings, to show how you give your own.
:::

## Read the trace

Every executed step records its method, inputs, outputs, parameters and
decisions. `result.to_dataframe()` gives one row per step, and
`result.print_trace()` prints every record in full.

```{code-cell} python
result.to_dataframe()[["Step", "Status", "Method", "Output Curves"]]
```

The `rw` row shows which Rw source was used:
`user_provided_temperature_profile_corrected`, the Rw you gave, corrected from
230 degF to the temperature of each depth.

## Summarize the Brent Group

The log starts in shallow, poorly consolidated rock that can pass the pay
cutoffs (the clay, porosity and saturation limits). One Rw also serves the whole
well, while shallow formation waters are usually fresher than the reservoir's,
so shallow sections that no lithology code covers can be flagged as marginal
pay; {ref}`review-pay` explains this limitation. Summarize the interval you
are evaluating, not the whole well.

```{code-cell} python
result.summary()["net_thickness"]
```

```{code-cell} python
brent = result.summary(top=2841, bottom=2958)
brent["net_thickness"]
```

Of the 51 m of net pay in the whole well, 25 m lies in the Brent Group. The
statistics of the interpreted curves over the same window:

```{code-cell} python
result.statistics(top=2841, bottom=2958)
```

## Plot the interpretation

Passing the result plots the interpreted well. The `petrophysics` template adds
porosity, saturation, permeability and pay-flag tracks.

```{code-cell} python
pp.plot(result, top=2841, bottom=2958, template="petrophysics",
        formation_name="Brent Group")
```

## Export

`to_las` writes every curve plus the settings used, as `PP_*` parameters in the
LAS header. `to_json` writes the complete trace.

```{code-cell} python
result.to_las("30_6-5_interpreted.las")
result.to_json("30_6-5_trace.json");
```

The `PP_*` parameters written to the LAS:

```{code-cell} python
pd.DataFrame(result.provenance_parameters())
```

## Several wells

`pp.read` and `pp.interpret` take a folder, a file pattern (`"logs/*.las"`) or a list of files as well as one file.
A multi-well tops table is read once and each well keeps its own picks; every well is interpreted with the same settings, its own header values filling the rest.
One table covers all the wells, and `to_las(folder)` and `save_plots(folder)` write a file per well.
A well that fails is named in `results.failed`; the others are interpreted.

```{code-cell} python
wells = pp.read([pp.sample(name) for name in ["HW-3", "HW-4", "HW-5"]],
                formations=pp.sample("formations_costa"))
results = pp.interpret(wells, lithology_domain="carbonate")
results.summary()
```

```{code-cell} python
results.formation_summary()
```

`pp.plot` on several wells draws each well's own log plot, one after another; `.save(folder)` writes a file per well.
With `curves=`, it draws one figure instead: a track per curve, with every well's curve in it.

```{code-cell} python
pp.plot(results, curves=["GR", "RDEP", "PHIE", "SW"])
```

## Next

{doc}`own_data`: read your own CSV logs, attach formation and lithology tables, and check that their depths agree.

The {doc}`complete tour <tutorials/complete_tour>` runs every capability of the library on bundled data in one notebook, and the {doc}`30/6-5 case study <tutorials/case_studies/cs1_full_formation_evaluation_30_6-5>` evaluates this well step by step.
