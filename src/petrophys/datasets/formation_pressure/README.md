# Formation pressures: two wells, CSV and LAS

Source: Norsk Hydro, *Final Well Report, well 35/11-13*, NH/OD-B-4522/05,
Rev 1, 13 December 2005. Norwegian Offshore Directorate wellbore ID 5063.
Copied from the user's North Sea Reports collection, file
`5063_35_11_13_COMPLETION_REPORT_3e589220b5.pdf`. The manifest pins its SHA-256.
Third-party attribution and rights are recorded in `THIRD_PARTY_DATA.md`.

**The bundled LAS files are Petrophys exports of report data, not native MDT
acquisition files. FLUID codes and USE_FIT flags were added for a labeled-fit
example. Native MDT data need not include any fluid classification.** The new
unlabeled API below accepts pressure and TVD with measurement QC only.

## Files and units

- `35_11_13_mdt_points.csv`: all 29 tests in Table 3.7.4, original PDF page 54
  (printed A-52). Numeric values come directly from the table underlying the
  pressure plot, avoiding graphical digitization. Includes remarks and zones.
- `35_11_13_mdt.las`: the same stations in
  [CWLS LAS 2.0](https://www.cwls.org/wp-content/uploads/2014/09/LAS_20_Update_Jan2014.pdf),
  with `STEP=0` for irregular sampling and `NULL=-999.25`. No resampling.
  `DEPT` is **TVD below MSL, positive down, metres**; `MD` retains measured
  depth from RKB separately. `PRES` is formation pressure in **bar**, not the
  mud pressure. The report does not explicitly label absolute versus gauge
  pressure; all formation pressures use the same CQG quartz gauge.
- `35_11_13_pressure_plot.png`: Figure 3.7.5, original PDF page 55 (A-53).
- `35_11_13_pressure_source.pdf`: original PDF pages 44, 45, 54, 55, retaining
  the interpretation, quality discussion, fluid samples, pressure table and plot.
- `source_manifest.json`: source and derivative hashes, page mapping and
  reported contacts/gradients.

LAS interpretation codes: `FLUID` 1=gas, 2=oil, 3=water; `QC` 0=ok,
1=low mobility, 2=dry test; `USE_FIT` 1=selected for oil-water fitting.
The dry test remains in both formats with missing formation pressure/mobility.
CSV `FLUID` and LAS `FLUID` are **added interpretation labels** based on the
report's gas/oil/water intervals, not direct fluid samples at every station.
The labels preserve the report's GOC at 3098.5 m MD and FWL at 3138.0 m MD;
the **fit uses the tabulated TVD**, never MD or a constant MD-to-TVD shift.

## Reproducing the interpretation with supplied report labels

```python
import petrophys as pp
from petrophys.interpretation.saturation.height import read_formation_pressure
from petrophys.visualization import create_formation_pressure_plot

data = read_formation_pressure(pp.sample("35_11_13_mdt_points"))  # CSV
# Alternatively: read_formation_pressure(pp.sample("35_11_13_mdt"))  # LAS
fit = data.fit_fwl()
print(f"FWL: {fit.fwl_depth:.1f} m TVD below MSL")
fig = create_formation_pressure_plot(data, fit)
```

Unweighted least squares through the 10 oil and 9 water tests marked `ok`
gives **3113.726 m TVD below MSL** (pressure **316.487 bar**), with gradients
approximately **0.06524 bar/m oil** and **0.09626 bar/m water**.
The reported FWL is **3112.6 m TVD below MSL**, with gradients 0.0642 and
0.0951 bar/m. The roughly 1.1 m difference reflects fitting/QC choices: the
report describes placing lines through higher-mobility points, not this exact
OLS selection. We do not tune the fit to force agreement.

This is a reproduction check of a known interpretation; fluid-branch labels
use report context, so it is not blind or independent FWL validation. Low
mobility can cause supercharging. Retain excluded points in plots for QC.
Do not mix different reservoirs or acquisition pressure references, use MD in
place of TVD, or equate pressure FWL with a saturation-defined oil-water contact.
The report's fluid scanning found water already at 3111.6 m TVD MSL, above its
pressure FWL; this illustrates the difference between mobility and equilibrium.

Tutorial 15 section 3b calls the library reader, fitter and plotter. It never
loads a PDF or source image. Report comparisons live in the tutorial,
not in the interpretation module. Its separate HW-3 saturation-height example
retains an assumed HW-3 FWL; another well's contact must not be applied to it.

## Second well: 35/11-10, Fensfjord Formation

Source: Norsk Hydro, *Final Well Report, well 35/11-10*, 12 December 1997,
`3042_35_11_10_COMPLETION_REPORT_0855941dce.pdf`, from the same local collection.
Figure 8.2 (PDF page 38, printed page 28) plots MDT station crosses and reports
FWL **2290 m TVD MSL**. Pages 34-35 describe the formation and fluid interpretation.

- `35_11_10_fensfjord_mdt_points.csv` and `35_11_10_fensfjord_mdt.las`:
  ten clearly identifiable station crosses (four oil, six water), digitized from
  the scanned plot. These are a subset of measured stations, **not samples of
  the fitted gradient lines**. Scatter is retained. Depths are rounded to 0.1 m,
  pressures to 0.01 bar; these digits do not imply acquisition-level accuracy.
- `35_11_10_pressure_source.pdf`: source pages 34, 35 and 38, retained for audit.
- `35_11_10_source_manifest.json`: source/derivative hashes, original page
  coordinates, render scale, every picked pixel and four axis calibration points.
  An affine transform corrects the slight scan skew. Picking precision is
  estimated at three pixels (~0.17 m and 0.03 bar), excluding scan distortion.

With the supplied report labels, `read_formation_pressure(...).fit_fwl()` gives **2287.335 m TVD MSL**,
**2.665 m shallower** than the report, with oil/water gradients
**0.06405/0.09860 bar/m**. The figure's gradient-equivalent densities are
0.68/1.02 g/cc. Plot resolution, rounding, station selection and original
interpretation choices affect this comparison. No points or fit parameters
were adjusted to reproduce the contact. The five-metre regression-test
tolerance is a reproduction check, not a claimed prediction uncertainty.
Fluid labels use report interpretation: this is not a blind validation.

## Unlabeled station pressures: the default tutorial workflow

```python
import pandas as pd
from petrophys.interpretation.saturation.height import read_formation_pressure
from petrophys.visualization import create_pressure_contact_plot

stations = pd.read_csv(pp.sample("35_11_13_mdt_points"))
stations = stations.drop(columns=["FLUID", "USE_FIT"])
stations["USE_FIT"] = stations.REMARKS.eq("ok")  # QC, including gas stations
data = read_formation_pressure(stations)
result = data.infer_contacts()
print(result.status, result.fwl_depth)
fig = create_pressure_contact_plot(data, result)
```

Your own unlabeled CSV or station LAS can be passed directly to the reader.
`infer_contacts()` ignores fluid labels; selection must be measurement QC,
not a previous contact interpretation. This example admits 21 good measurements
and gives a two-branch oil/water candidate at **3113.174 m TVD MSL**. It does
not resolve the two shallow gas stations, which span only 0.9 m, separately.
For 35/11-10, removing labels yields an intersection near **2289.134 m**, but
several possible phase sequences: the result is **ambiguous**, `fwl_depth=None`.
These are unlabeled reproduction checks with known data-quality limits, not
general validation on unseen fields. No reported contact enters inference.

The method compares contiguous one/two/three-gradient models using Huber IRLS.
Its score is `2 * Huber loss + (3*k - 1) * log(n)`, a penalized fit score, not
a probability or formal BIC. Segmentation precedes fluid typing. Gradients
convert by `density_g_cc = gradient_bar_per_m / 0.0980665`; ordered phase
hypotheses are then tested. The lowest group is not simply named water.
Unstable intersections, single gradients, conflicting density hypotheses and
near-equal scores across branch counts do not produce a definite FWL.

Defaults are analysis assumptions, not universal vendor specifications:

- Pressure error scale: 0.05 bar equivalent. Supply `pressure_noise` in the
  input pressure unit; it affects fitting, selection and sensitivity.
- At least three distinct depths and one metre equivalent per branch.
- Overlapping density bounds: gas 0.02-0.50, oil 0.45-1.05 and water
  0.90-1.30 g/cm3. `fluid_density_ranges` accepts reservoir PVT bounds and
  can restrict phases where independent evidence establishes phase presence.
- Contact intervals are conditional +/-2-sigma sensitivity to pressure error
  and residual scatter. They exclude depth, segmentation, phase-prior and
  connectivity uncertainty. Suspect points are flagged, not deleted.

This assumes one prospective connected reservoir and at most one leg of each
fluid. Inconsistent pressure jumps can be flagged, but arbitrary hydraulic
compartments cannot be discovered or their connectivity proved. Samples,
downhole fluid analysis and conventional logs remain independent evidence.

## What a real MDT acquisition looks like

All 35 LAS files in the inspected public
[JAMSTEC C0020A XLD archive](https://www.jamstec.go.jp/sio7/well-logging/337/C0020A/ORIGINAL/DATA/337_C0020A_Run4_MDT_XLD_pretest_LAS.zip)
have these channels:

| Mnemonic | Unit | Measurement |
|---|---|---|
| TIME | S | Acquisition time (index) |
| PQDVP1 | M | Probe depth; not automatically TVD |
| PQQT1 | DEGC | Quartz-gauge temperature |
| PQQP1 | PSI | Quartz-gauge pressure |

No fluid code is present. This example is not a universal curve schema:
mnemonics and available modules vary. The
[source report](https://www.jamstec.go.jp/sio7/well-logging/337/C0020A/ORIGINAL/DOCUMENTS/337_C0020A_Run4_MDT_XLD_pretest_report.pdf)
separates formation pressure, buildup pressure, mud pressure, mobility and QC.
Taking the last sample blindly can pick mud pressure after probe retraction.

[SLB's workflow](https://www.slb.com/products-and-services/delivering-digital-at-scale/software/techlog-wellbore-software/techlog/techlog-reservoir-engineering/techlog-formation-pressure)
separates pretest pressure picking/QC from gradient/density/contact interpretation.
`read_formation_pressure` handles station tables. For raw time-indexed LAS,
use `read_mdt_pretest(path).pick_pressure()`. This automatically detects drawdowns
and tests pre-retraction buildup windows for stability and repeat agreement.
It supplies a numerical `pressure` only when the configured checks pass;
otherwise it returns review flags and a separate diagnostic candidate.

### Native C0020A fixtures and external validation

The 35 `c0020a_mdt_*.las` files are unchanged source bytes (filenames shortened)
from the archive above, recorded by Schlumberger for JAMSTEC, Expedition 337,
Run 4, September 2012. Source headers and attribution are retained. These are
third-party field deliveries; inclusion does not relicense them as Petrophys
code. `c0020a_mdt_manifest.json` records upstream member names, URLs and SHA-256
checksums. `c0020a_mdt_pretest_report.pdf` is the unchanged associated report.

`c0020a_mdt_report_reference.csv` records the **Formation Pressure** and
**Pretest Status** from each individual test page (PDF pages 14-48), plus probe
MD and file ID for matching. Formation pressure stays null for tight/lost-seal
tests: their last buildup values must not be mistaken for valid formation
pressure. No report pressures, statuses, fluid assignments or contacts are
arguments to the library picker. Tutorial 15 and the regression test compare
with the report only after running on raw curves.

With default `MDTPickingConfig`, 22/35 traces produce accepted stable-plateau
proposals; maximum absolute difference from published formation pressure is
0.149 psi. Both tight tests (086/112) and lost seals (105/118) require review.
Nine report-valid traces (070/085/087/095/096/097/113/114/115) also require review:
repeat disagreement, residual drift, an insufficient window or multiple mud
returns. A vendor's valid-test classification need not imply the stricter
plateau requirements are met. These thresholds are configurable interpretation
assumptions, not instrument accuracy. This is one field delivery, not a broad
validation population.

Supported scope: one overbalanced stationary probe setting per trace, bracketing
mud baselines, and potentially repeated drawdowns. The picker does not fit
transient asymptotes, correct supercharging, decode valves, or establish seal
integrity from pressure alone. It does not handle arbitrary pumpouts,
underbalanced tests or multistation acquisitions. Time must increase; missing
or gapped samples trigger review. Review results never supply a usable pressure.
Plots made with `create_mdt_pretest_plot` show original pressure samples and
the library's actual QC windows, not images extracted from a report.

For subsequent gradients, `formation_pressures_from_mdt` requires explicit TVD
depths and a common datum unless picks already carry verified TVD metadata.
Probe MD is never automatically TVD. Select a connected reservoir and common
pressure reference before inference. These C0020A fixtures validate picking;
they do not establish an oil/water contact across the whole borehole.

## Your own pressure data

The reader accepts CSV paths, LAS paths, pandas DataFrames and `lasio.LASFile`
objects. CSV names such as `TVDSS_M`, `TVD_MSL_M`, `TVD_RKB_FT` and `PRES_BAR`
encode units/datum. For other column names, specify the mapping and metadata:

```python
data = read_formation_pressure(
    "my_pressures.csv",
    depth_column="vertical_depth", pressure_column="formation_pressure",
    fluid_column="fluid_code", use_column="accepted",
    depth_unit="m", pressure_unit="bar", depth_reference="TVDSS",
    fluid_map={1: "oil", 2: "water"},  # use YOUR delivery's codes
    csv_kwargs={"sep": ";", "na_values": [-999.25]},
)
fit = data.fit_fwl()  # hydrocarbon="gas" for gas/water
```

Supported pressure units: bar, psi, kPa, MPa; depth: m, ft. Outputs retain input
units. LAS reads curve units, `DREF` and optional `FLDMAP` headers; otherwise
supply metadata explicitly. `DEPT` alone does not establish TVD. Numeric fluid
codes require a map; textual labels are oil/gas/water. `USE_FIT` accepts 1/0 or
true/false. Without a selection column, all finite stations in the chosen fluid
pair enter the fit. Missing values, unselected stations and other known fluid
branches stay in `data.points`; finite excluded stations appear on the QC plot.

Select one connected reservoir and a consistent pressure reference yourself.
For unlabeled stations, `.infer_contacts()` proposes fluid sequences from
gradients and exposes ambiguity and suspect points. Pressure alone does not
determine a saturation-height function: that still requires
capillary-pressure/rock calibration and fluid properties.
