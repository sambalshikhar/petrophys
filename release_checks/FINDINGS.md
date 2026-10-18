# Release audit findings

The audit is in progress. Passing automated checks below does not approve the
remaining module/equation or figure review rows in the Excel workbook.

## Method decisions (18 October 2026)

The open method questions were decided in [METHOD_DECISIONS.md](METHOD_DECISIONS.md):
the 33 items that change results were implemented, each gated so the benchmark
packages stay byte-identical, with regression tests in `tests/test_decisions_*.py`;
cosmetic, policy and architecture items are listed there as deferred, and the
licensing and public-URL items remain the maintainer's. Among the implemented
changes: explicit settings now win over header values (rm, phi_cl, measurement
temperatures, zone-only parameters stay in their zone); a measured TEMP log is
the temperature source; PERM_CC is written only inside the plugs' porosity
range; Dual Water uses the Clavier-Coates-Dumanoir form; the regional shaly-sand
models return NaN where they have no solution; absolute overburden and
thickness-weighted HCPV in the summaries; AT90 ranks first for RDEP; core
saturations in percent are converted to v/v; evaporites use their log density
response; core grain density decides the matrix; DCAL has its own QC rule and
value-range QC is unit-aware. Checks after the changes: full suite 1,746 passed;
all 30 notebooks re-executed (narrative updated where the numbers changed); the
54-well regression gives processed LAS identical to the deliveries and contexts
byte-identical to the pre-fix run.

## Second audit round (17 October 2026)

Scope: every row of the workbook's scientific-review (2,086 callables, 930
expressions), formats/routes, settings, public API, MCP/CLI/security, scripts,
datasets (including one or two Taranaki and Teapot wells), rights/packaging,
documentation, performance, all 30 notebooks and all 152 figures, run against
the working tree merged with `origin/dev` 22f668b. Reviewer results are in
`Q:` scratch evidence (`science/*.json`, `review/*.json`).

- **Blockers fixed** (each confirmed by an independent verifier):
  - Merged DLIS reads interpolated across absent-value runs, which fabricated
    68 m of data on the Daydream fixture. Absent runs now stay NaN.
  - NMR porosity in `m3/m3`, `ft3/ft3`, `dec` or `CFCF` was read as pu, which
    made permeability 1e8 too low.
  - Percent core porosity was never converted to fraction, which corrupted
    PHIE_CC and the poro-perm fits.
  - Standardization dropped or mislabelled curves that mapped to the same
    standard name (for example AT90 and HD1_PPC1 were lost). Both curves are
    now kept, with a warning.
  - Simandoux returned Sw = 0 wherever porosity was 0. It now uses the stable
    form, which gives the Rcl/(Vcl*Rt) limit.
  - The default Rw route accepted an SP estimate stuck at the 0.01 ohm.m clip,
    which inflated 30/6-5 net pay 16x. That estimate is now rejected, and
    in-range SP estimates warn.
- **Clear defects fixed:** 204 across I/O, units, NMR/geochemistry, core,
  electrical/permeability/temperature, lithology/imaging, the workflows,
  evaluation, reporting/QC, the model, visualization, MCP/CLI and packaging.
  Each has regression tests in `tests/test_fixes_*.py`.
- **Notebooks:** all 30 had text corrected to match their outputs (real data
  only) and were re-executed.
- **Bundled data:** the 30/6-5 header now carries RW 0.070 ohm.m at RWT 230 degF
  (NPD), in place of RW 0.7 with no temperature.
- **Benchmark reproduction is unchanged:** the context profiles run inside
  `petrophys._internal.benchmark.historical_benchmark()`, which keeps the
  historical behaviour. On all 54 wells (32 GeoLink, 17 COSTA, 5 Danish) the
  processed LAS are identical to the deliveries, and the contexts are
  byte-identical to the pre-fix run. Against the delivered contexts, only the
  documented limits differ: unseeded ELECTROFACIES statistics, and the
  SO/note/bit-size metadata of 15/9-17 and 34/10-23.
- **Installs:** 10/10 clean Windows installs (wheel and sdist, Python 3.9-3.13)
  passed before the fixes. Building an sdist on Windows needs a short build path
  (under about 150 characters) because of the path-length limit.
- **Maintainer decisions still open:**
  - The 79 scientific-method questions and 47 unfixed items listed in
    [METHOD_QUESTIONS.md](METHOD_QUESTIONS.md).
  - Permission documents for GeoLink-derived data and model, Norsk Hydro,
    RealPore and EpsLog. The maintainer states the permissions are held; the
    documents are still to be added.
  - Teapot (DOE/RMOTC) and Taranaki (CDLA-Sharing-1.0) licence terms before
    bundling any of their wells.
  - Public repository/install route for the documentation links.
  - The Linux/macOS matrix.

## Corrected during the audit

| Finding | Observed behavior | Correction and evidence |
|---|---|---|
| Scalar/array disagreement in Timur, Coates-Denoo and Wyllie-Rose | For positive fractions phi=0.0005 and Swi=0.005, scalar paths imposed floors while arrays retained the inputs. Timur returned 5.4142575e-6 mD for scalars and 1.02580998e-6 mD for the same array values. | Apply the same nonpositive-input fallback to both shapes and retain positive input fractions. Independent cases and `test_release_scientific_contracts.py` verify the equations and shape equivalence. |
| Unknown measurements becoming zero porosity/clay volume | Missing density/sonic values with identical matrix/fluid endpoints returned zero porosity. Equal GR baselines returned clean-rock VCL, including for missing GR. | Undefined contrasts and missing inputs return NaN. Tests also verify that scalar APIs retain scalar outputs. |
| Infinite resistivity becoming zero water saturation | Archie returned Sw=0 for infinite Rt. | Preserve unknown/nonfinite measurements as NaN. |
| Pressure tutorial selected an interpreted fluid mask | The unlabeled contact example removed fluid labels but retained an oil/water fit mask. Its accompanying text described 21 good measurements. | Rebuild selection from the LAS measurement-quality curve (`QC==0`); include all 21 good measurements. Remove text about deleted raw MDT fixtures and re-execute the notebook. |
| Invalid orientation notebook metadata | Markdown cells contained `outputs` keys, which nbformat rejects. | Remove code-only metadata from markdown cells; fresh execution now proceeds. |
| Phillips verification used the wrong bin partition | The script used the partial-bin integration default against source summaries generated by a sharp below/at-32-ms partition. MBVI closure failed by 1.638 pu. | Request `partial_bins=False` for source-summary closure; the full verification script now passes. |
| Benchmark-script documentation test failed | The parser still accepted the documented invocation, but the module docstring no longer contained the example required by a test. | Restore a concrete invocation without workflow-history wording. |
| Ambiguous GR-normalization labels | Histograms said `0-100` even though percentile normalization preserves values outside the reference anchors. | Label P5=0/P95=100 explicitly and explain that values are not clipped; re-execute the affected lessons. |
| NMR tutorial overclaimed fluid and processing interpretation | Text described cutoff/permeability processing as necessarily downhole and treated low GR/caliper flags as unique proof of spike causes. | Describe the delivered processing, fluid/acquisition assumptions and non-unique QC indications; cite SLB's processing and relaxation references and re-execute the lesson. |
| PHIE_CC incorrectly depended on core permeability | Dynamic and standard workflows required KH_CORE as well as PHIC; a later permeability step could also erase the generated porosity correction. | Fit porosity independently using PHIC/CPOR and log porosity; preserve that result through permeability calculation. Missing core permeability does not create PERM_CC. Well 35_9-2 now exports 12,315 finite PHIE_CC samples and no PERM_CC; all 930 context files generate successfully. Evidence: `outputs/phie_cc_fix/verification.json`. |
| Valid reversed porosity endpoints rejected during audit | An initial audit guard required positive endpoint contrast, dropping coal sonic porosity. | Require nonzero contrast instead. Independent mass/time balances pass for either contrast sign. Fresh PHIS/PHI_SEC arrays match the pre-audit code exactly for the three baseline sample wells. Evidence: `endpoint-baseline-refreshed.json`. |
| Long Windows context paths failed during staged export | The Daydream category package exceeded the Windows path limit in a deeply nested output directory. | Use extended Windows paths for filesystem operations while preserving public filenames. Regression verifies long single-file/package exports, existing-output protection and atomic failure. All 29 context API tests pass. |
| NMR figure omitted depth units | Daydream's three-track plot showed depth ticks without a depth-unit label. | Add the LAS depth unit explicitly. Fresh execution and full-resolution reinspection confirm the corrected label. |
| Saturation-height figure mixed units on one scale | PHIE_CC and PERM_CC shared a logarithmic track without a common unit. | Plot porosity and permeability on separate tracks with their own scales and units; add depth units. Fresh execution and full-resolution reinspection confirm the correction. |

## Completed execution checks

- Refreshed full optional-dependency suite: **1,445 passed** on Windows/Python
  3.11, including the PHIE_CC and long-path fixes (`pytest-full311-refreshed.xml`).
- **34/34 independent numerical spot checks** pass, including NMR bin-volume
  balance and a constructed isotropic elastic medium. They do not approve every
  equation or field application.
- **30/30 notebooks** pass the complete fresh-kernel run after the latest fixes
  (`notebooks-refreshed.json`). The strict Sphinx build passes, including
  execution of the capabilities guide and rendering the revised tutorial
  figures (`docs-build-refreshed.log`, `docs-build-tutorial-figures.log`).
- Fresh wheel and sdist build and pass metadata validation. **10/10 clean
  Windows installs** pass on Python 3.9-3.13; installed reports match the fresh
  artifact hashes. These are smoke tests, not full-suite runs on every
  interpreter.
- All **766 declared exports** import successfully from the fresh installed
  wheel. Optional-dependency installed-wheel tests pass **255 cases**; two
  optional external-DLIS context cases initially skipped and subsequently both
  passed against the real Daydream file. The installed package import path was
  verified explicitly (`pytest-installed-wheel-optionals-refreshed.xml` and
  `pytest-installed-dlis-context-refreshed.xml`).
- Epslog hash/calculation checks pass, including byte comparisons with the
  eight original LAS files. Phillips checks pass within their recorded scope.
- The earlier dependency scan returned no known vulnerabilities. Contextual
  review of the fresh static scan closes five specific findings; ten remain
  pending, chiefly silent exception/fallback reporting. This does not constitute
  complete security approval (`checks-static-context.json`).
- **152 freshly generated figures** decode successfully and were inspected on
  contact sheets. Full-resolution scientific/visual approval remains pending.
- Repeated local performance measurements now include sampled process RSS
  (Python and native memory). Median read/interpret times are 0.184 s for HW-3,
  6.083 s for 30/6-5 and 2.147 s for the KGS sample, with peak observed process
  RSS of approximately 163/246/226 MB. Three sequential repetitions and 20 ms
  memory sampling are measurements, not cross-platform performance approval
  (`checks-performance-rss-refreshed.json`).

## Dataset-reference comparisons

Fresh computation from raw inputs completed for all **54 wells**: 32 GeoLink,
17 COSTA and five Danish. All **25,992 JSON files** parse strictly, and every
generated/reference filename matches. Source hashes and exact field-level
differences are recorded separately. These results supersede the earlier batch.

COSTA: all 17 wells produced 270 valid JSON files. **225 match exactly**;
the 45 numerical files differ only in ELECTROFACIES statistics (286 numeric
fields). Reference clustering was unseeded; retain the exact comparison failure
and review this documented reproduction limit rather than treating it as a pass.

Danish: use the tops workbook stored with the reference delivery. The similarly
named Desktop workbook appends `FM.` to formation names and therefore changes
all output paths. With the matching reference inputs, all five wells produce
the same 156 filenames: **23 files match exactly**, 133 differ. All 152 numeric
differences are ELECTROFACIES statistics; other differences are computation-note
text/order, including the stated reservoir-screening cutoff. These differences
are recorded explicitly and are not silently excluded from exact comparison.

GeoLink: all 32 wells produced **25,566 strictly valid JSON files** (4,261 per
category). **2,840 files match exactly** and 22,726 differ. The principal
numerical change is the requested PHIE_CC correction:

- Nine wells contain finite PHIC but no finite KH_CORE: `16_2-11 A`, `16_2-6`,
  `25_10-8`, `34_10-35`, `34_11-2 S`, `35_11-11`, `35_4-1`, `35_9-2` and
  `35_9-7`. All nine now export PHIE_CC; none exports PERM_CC. Downstream
  saturation, empirical permeability, RQI/FZI, flags and completeness statistics
  change because calibrated porosity is available. Raw/core counts are in
  `porosity-only-wells-refreshed.json`.
- The earlier PHIS/PHI_SEC and VCALC/VDOLO/VQUAR statistics discrepancies are
  absent from the fresh exact comparisons. For example, Hordaland mean VCALC
  for `25_11-19 S` now matches the reference at 0.096.
- SO statistics are absent for the two already documented reproduction-limit
  wells (`15_9-17`, `34_10-23`); ELECTROFACIES statistics also differ.

Other differences include computation notes and availability/QC metadata. The
three fresh field-level reports are `context-differences-geolink-refreshed-1.json`
through `-3.json`; the statistics overview is `geolink-refreshed-statistics.json`.
Generation results are consolidated in `checks-datasets-refreshed.json`.
All evidence paths above are under `outputs/release_audit/evidence/`.
Exact reference equivalence remains failed; intentional scientific corrections
do not silently waive the comparison.

## Required work still open

- Complete independent scientific review of all workbook rows. The 34 spot
  checks verify stated points/invariances only, not every equation or field
  application. Field-unit J and hydrostatic constants are compared against SI
  with explicit tolerances for their documented approximations.
- Complete individual review of fresh figures. Contact sheets assist browsing;
  thumbnail inspection alone does not approve axis units or interpretation.
- Review the documented reference-comparison differences, including the
  intentional porosity-only correction and its downstream effects, remaining
  SO/electrofacies limits and availability/QC metadata. Do not claim unchanged
  benchmark results.
- Resolve scientific input limits before claiming validated field workflows:
  COSTA plug Pc pressure units/fluid conversion and plug-to-well ties are
  unverified; the Phillips bin example lacks traceable field-acquisition
  provenance; Epslog/RealPore core permeability units and depth calibration
  remain unconfirmed, and scanner strength is not independently validated plug
  UCS. These are separate from the deferred redistribution checks.
- Execute the prepared Linux/macOS artifact matrix. Local Windows checks do
  not establish portability to those operating systems.
- Agree performance limits and review repeated latency/RSS measurements;
  cross-platform load behavior and brief allocation peaks remain unverified.
- Review the missing-clay-volume screening convention: some diagnostic/net-pay
  paths warn and substitute VCL=0, which can overestimate flagged reservoir/pay.
  Passing those contract tests does not independently justify that assumption.
- Complete the deferred data/model redistribution checks, Taranaki/Teapot
  verification, version selection and publishing setup.
- Rehearse installation from TestPyPI and enforce artifact identity through
  publication. The existing TestPyPI-upload job is not installation evidence.
  The release approval gate currently refuses upload because required evidence
  and permissions are pending/deferred.

The two medium-severity static-scan findings have scoped dispositions: the CLI
bind path has an authentication guard, and the flagged `Select unique track
names` string is an error message rather than a SQL query. The ten remaining
low-severity findings need error-handling/provenance review; programmatic HTTP
deployment boundaries also remain open. These dispositions do not replace the
MCP security tests or full security review.
