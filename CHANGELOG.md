# Changelog

Release notes for Petrophys: what each version of the library does.

## 3.0.0 (unreleased)

First public release. Petrophys reads well logs, runs a traceable
petrophysical interpretation, evaluates net pay and reservoir quality, plots
and exports the results, and writes compact JSON context for LLM agents. Two
calculation profiles are available: `"current"`, the maintained
interpretation and the default, and `"geolink"`, the calculations of the
GeoLink eval context benchmark package. Context profiles regenerate the
published GeoLink, GeoLink training, COSTA and Danish context packages from
their raw logs.

### Calculation profiles

- `DynamicWorkflowConfig.calculation_profile` chooses the calculations.
  `"current"`, the default, is the maintained interpretation described under
  "Interpretation". `"geolink"` keeps the calculations behind the GeoLink
  eval context benchmark package, so the package can be regenerated from its
  raw LAS files. The setting's documentation lists every difference between
  the two.
- The profile is selected with `calculation_profile="geolink"`,
  `petrophys-mcp serve --calculation-profile geolink`,
  `PETROPHYS_CALCULATION_PROFILE=geolink`, and `class_tables="geolink"` for
  the diagnostics analyzers.
- The GeoLink package was made in two runs: its main run interpreted 30 of
  the 32 wells, and a second run interpreted 15/9-17 and 34/10-23 and wrote
  the context of every well. Under `"geolink"`, interpreting a well
  (`petrophys.interpret`, `run_dynamic_workflow`, `DynamicWorkflow`, the
  zoned batch runner, the MCP server) uses the main run's calculations,
  except for 15/9-17 and 34/10-23, which get the second run's. Either well is
  recognised by its UWI, its WELL header or, in the zoned batch runner, its
  LAS file name, compared without case or surrounding spaces and with `/`
  read as `_`; `15/9-17 A` is another well. Writing context under
  `"geolink"` uses the second run's calculations for every well. The
  workflow trace names the calculations a well got in
  `config.benchmark_calculations`: `"main_run"` or `"rerun"`.
- The main run's calculations take clay porosity from the mean PHIT of every
  shale-lithology sample; read PHIE and PERM rather than the core-corrected
  PHIE_CC and PERM_CC, with no PHIT fallback; use one Archie coefficient set
  from the lithology domain rather than per-depth values; use the SXO
  quick-look relation only as a whole-curve fallback; take the invasion ratio
  from the first flushed-zone or shallow resistivity present, without the
  medium-resistivity proxy; compute MHI wherever SW and SXO are finite; and
  write a duplicate BADHOLEFLAG curve.
- `petrophys.interpret`, `export_context` and the MCP server use `"current"`
  unless told otherwise; `petrophys-context` does too unless a context
  profile is selected. To apply the GeoLink calculations to other wells, pass
  `workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True)`
  to `export_context`: the package picked SWIRR_APP from the reservoir flag
  already on its interpreted LAS files.
- `petrophys-mcp serve --calculation-profile {current,geolink}`, or the
  `PETROPHYS_CALCULATION_PROFILE` environment variable, sets the profile of
  the MCP server's interpretations whose settings name none, so that an
  evaluation harness can pin `"geolink"`. A profile in a tool's settings
  wins, `petrophys-mcp check` reports the active one, and `create_server`
  takes it as `calculation_profile`. Only the MCP server reads the variable;
  the library ignores it.
- `config_schema()` lists `profile_defaults` for the settings whose default
  depends on the profile: `standardize_input`, `vcl_cutoff` and `sw_cutoff`.
- Context exports, and the hydrocarbon analysis and invasion statistics of
  the workflow, classify with the class tables of their profile: `"current"`
  with the canonical tables of `petrophys.model.CLASS_TABLES` (see
  "Classes and reservoir quality"), `"geolink"` with the eval package's
  tables, rules and threshold texts. The diagnostics analyzers and
  `run_comprehensive_analysis` take the choice as `class_tables`,
  `"current"` by default.

### Benchmark context packages

- `petrophys.interpret(path, context_profile=...)` interprets one well of
  the GeoLink, GeoLink training, COSTA or Danish package from its raw LAS
  exactly as the package's context does: the profile's preparation and
  calculations, then the context's value-range QC and dependent-curve
  recomputation. The statistics of `result.well` over a context group's
  depths are that group's statistics. The Danish profile also takes
  `formations`, its tops workbook. `result.steps_executed` records every calculation: the
  profile's preparation (COSTA's clay volume, carbonate volumes and
  lithology, and permeability under the workflow's step names; the
  preparatory interpretation's RESFLAG, the Danish input preparation and
  input standardization as `preparation_*` steps, starting with
  `preparation_input_reading`: the renames, unit conversions and range
  screening of reading the raw LAS), the workflow, and the context's export
  finalization, QC and recomputation (`context_*` steps). Every curve of
  `result.well` is an unchanged raw curve or the output of a recorded step. The same is available as
  `petrophys.reporting.context_profiles.interpret_with_context_profile`.
- `petrophys.reporting.context_profiles.context_group_statistics(result, top, bottom)`
  computes one context group's statistics from such a result, per category
  file, including the curves each category's analysis derives within the
  group (indicators, gradients, completeness, and MHI, BVW, RQI, FZI or
  PHI_NORM where the group has none). `context_group_record` returns the
  group's six context documents in full, notes included (saturation routing,
  Archie coefficients per lithology, Rw, Rcl, SXO route, cutoffs per
  lithology family), as the package writes them; the exporter writes the same
  documents.
- `result.group_records()` gives each formation and lithology group of an
  interpretation its own record, as its context files hold it: saturation
  routing and reasons, Archie coefficients per lithology, Rw, Rcl, SXO route,
  cutoffs per lithology family (`documents=True` for the six documents). The
  groups are those of the context package; the exporter and this method share
  the grouping and the documents.
- A context package interprets each well as `petrophys.interpret` does with
  the same inputs: its formation tops are attached for per-formation core
  calibration, header Rw, Rmf, Rm and BHT stay on the well (recorded as LAS
  metadata), TD comes from the header, and the hydrocarbon-correction step is
  left to the workflow. Benchmark profiles keep their packages' handling.
- `pp.plot(result)` shows the interpretation it is given: the tops,
  lithology intervals and lithology codebook passed to `pp.interpret`, and
  the curve values the interpretation used (units converted for display, no
  value screened again). The automatic layout draws the mineral volumes
  whenever they exist, and RQI/FZI. The computed temperature profile and
  Rw/Rmf tracks are optional: `viewer.diagnostics["optional_tracks"]` lists
  them and `tracks=` draws them (a logged temperature keeps its track).
- Plots of a raw well show every logged curve: SP joins the gamma track,
  GAM, MTEM, core porosity and permeability (CORE_POR, CPERM, ...) and vendor
  porosities (DPOR, SPOR, drawn in v/v when given in percent) fill their
  tracks, and curves no track takes go to up to four generic tracks; the
  viewer warns about any it still leaves out, and about curves whose values
  all lie outside their plausible range. A curve drawn under an alias keeps
  its own name in the header (DPOR, not PHID). Generic tracks leave out
  interpretation intermediates (VCL_GR, PHI_SEC, RESFLAG, ...), the
  lithology code curve and exact copies of a drawn curve (a working alias
  such as DTC for DT); a curve is not drawn twice under another family's
  alias (NPHI as PHIN). Core grain density joins the density track, core oil
  saturation and a vendor SW_INPUT the saturation track.
- The automatic layout scales SP to the well's own baseline (at least 100 mV
  wide), so an SP curve with a positive offset stays on scale (30/6-5 reads
  50-110 mV); FZI uses 0.01-1000 microns.
- Log display conventions in the automatic layout: on a well without a
  bad-hole flag (a raw well), washout shading where the caliper exceeds bit
  size; a flagged well keeps the bad-hole band across all tracks as its only
  hole-condition signal; density-neutron crossover (yellow) and shale
  separation (grey) on compatible sandstone scales; neutron dashed; medium,
  shallow and flushed-zone resistivity dashed, dotted and dash-dotted; sonic
  in its own SONIC track (DTC, DTS); BVW on the porosity scale, so PHIE to
  BVW shows hydrocarbon and BVW to zero water; VCL drawn as a line (clay
  volume is shaded in the mineral volumes track). Header height follows the
  number of curves, so labels do not overlap on short figures, and
  generic-track scales are rounded (60-180, not 68.3032-170.051).
- `pp.plot(..., curve_styles={"GR": {"color": "black", "right": 200}})`
  changes a curve's colour, scale, line, fill or marker without an XML
  template; unknown keys raise an error, names no track draws a warning.
- Mineral solver, checked against synthetic mixtures, core grain density and
  vendor lithology logs:
  - Without PEF, the clastic solve (version 1) takes quartz, calcite and
    dolomite from RHOB and NPHI; the sonic does not stand in for PEF, which
    would read most carbonates as quartz. The solve calls 74-89 % of the
    vendor-coded carbonates of 15/9-14 and 30/6-5 carbonate. Beside PEF the
    sonic row is kept.
  - PEF is set aside for a well when it reads like barite-weighted mud: in
    most clean porous samples quartz-like apparent matrix density with
    calcite-like apparent matrix U. On 30/6-5 the Brent Group sands then solve
    as quartz (88 % agreement with the vendor log) and the solved grain
    density is within 0.013 g/cm3 of core. The lithology step says so and the
    result carries a warning.
  - A sandstone-referenced neutron (NPSS, CNSS, NPHI_SS) is converted to the
    limestone scale the interpretation reads (first-order -0.04 v/v), with a
    warning.
  - Samples version 1 cannot solve for want of NPHI, where PEF is logged,
    are solved by version 2 (porosity solved jointly): 25/5-1, whose NPHI
    covers a short interval, has mineral volumes over 99 % of the well,
    agreeing with its vendor log on 90 % of those samples. A well with
    neither NPHI nor PEF skips the step and says it needs one of them.
  - The COSTA calibration solves its carbonates in closed form,
    `solve_carbonate_minerals(..., "costa")`, with the equations of the
    per-sample SLSQP solve in the COSTA notebooks: the two agree to 1e-7 v/v
    wherever the optimizer converges, and the closed form also solves the
    samples where it does not (HW-3: 39 samples, against 20 under the
    "geolink" calculation profile). The COSTA package keeps its own solve and
    rebuilds byte for byte. `costa_carbonate` holds the COSTA constants.
  - The COSTA calibration's pore fluid is 1.1 g/cm3, the value Craig
    Phillips's notebooks use in their calcite-dolomite solve (their FD = 1.19
    is reset to 1.1 before it runs, as the saved execution order shows), not
    1.19. At 20 % porosity the difference is worth about 13 % of the matrix
    between calcite and dolomite.
  - `rho_fl` set by the user reaches the COSTA calibration's solve. Left
    unset, `rho_fl` is the
    COSTA brine (1.1) when `carbonate_calibration="costa"` is chosen and
    1.0 otherwise, so porosity and the mineral solve read the same fluid;
    `result.config.rho_fl` holds the value used. `rho_cl` likewise reaches
    the COSTA solve (unset: its 2.52 illite).
  - The benchmark profiles keep their packages' solve.
- Values you set are used as given (normal runs; the benchmark profiles
  keep their calculations):
  - `rho_ma`, `dt_ma` and `nphi_ma` hold at every depth, also where a
    lithology log sets matrix values per group.
  - `salinity_ppm` wins over a header RW, and `rw_meas_temp_f` over a
    header RWT; the Rmf step ranks explicit settings first too.
  - `bht_f` and `surface_temp_f` win over an input TEMP log (kept as
    TEMP_LOG); zone runs place the BHT at the well's TD, not each zone's base.
  - The SP-derived Rw uses `sp_max` and `sp_min` as its clean and shale lines.
  - The clastic mineral solve uses `rho_fl` and `dt_fl` for its water
    point; core-matrix inference uses `rho_fl`. Waxman-Smits Qv takes the
    grain density of `matrix_type` when `rho_ma` is unset.
  - `apply_hc_correction=True` without an input SXO skips the step with a
    warning: the step runs before saturation, where an assumed SXO of 1
    would change nothing.
  - In zoned and batch runs a setting passed explicitly keeps its value over
    a LAS header value even when it equals the default.
  - GR baselines you set (`gr_min`/`gr_max`, header `GR_MIN`/`GR_MAX`, zone
    `GR_MIN_Z1`) are used as given; the 20/100 bound applies only to the
    baselines picked from the log (P5/P95 in the workflow, the log extremes in
    `VclModule`/`calculate_vcl`), as a guard against a nearly flat GR.
    `vcl_from_gr` uses the baselines it is given; `bounded_gr_baselines` is
    the bound. A clean pick above 30 API that the bound lowers warns (clean
    rock may be radioactive): on Cheal-1 the bound lowers the P5 of 64.8 API
    to 20, and its Mount Messenger sands, about 86 API, then read 62 % clay.
    The "geolink" profile bounds every baseline.
  - `NMRData.from_las(t2_cutoff=...)` beats a T2CUTOFF curve in the file.
  - `calculate_zone_averages(use_arithmetic=...)` is applied.
  - The reduced quartz-calcite fallback of the version 2 solve keeps the
    caller's `solver_options`.
  - `read_dlis` applies `standardize` to a single frame, and reads a frame
    given `step`, `step_strategy` or `channel_preference` through the merged
    path, which honours them.
  - `pp.plot` draws a lithology `fill_alpha` set in `curve_styles` as given.
  - `eaton_from_well` converts a TVD curve in its own unit and, offshore,
    takes TVDSS from `air_gap`.
  - `monte_carlo` keeps `low`/`high` on depth-correlated lognormal inputs;
    `read_uncertainty_table` finds a fixed row's value beside a `mean` column.
  - `MDTPickingConfig(max_gap_seconds=...)` is used as given (unset: 5 s or
    five sampling intervals).
  - `leverett_j`/`pc_from_j` use |IFT cos theta|, so a 140-degree
    air-mercury angle gives a positive J.
  - Unknown choices raise an error, with no fallback: `PorosityModule` and
    `VclModule` matrix, `sonic_water_saturation` `hc_type`/`lithology`/
    `kdry_method` (case-insensitive), `assign_flow_units` method and
    `fit_j_function` form. `n_units` above the distinct FZI values and a
    `min_thickness` under two samples warn.
- The same settings reach the calculation on every route (normal runs):
  - `run_dynamic_workflow`, `DynamicWorkflow` runs and the batch processor
    apply the LAS header settings (A/M/N, MATRIX, LITH, SAL, VSHE, ... as
    well as RW, BHT, RMF and RM) as `pp.interpret` does.
  - `pp.interpret(well, config)` lets header values fill the settings the
    config leaves at their defaults, as with keyword settings.
  - Changing `lithology_domain` on an existing config (`interpret(config=...,
    lithology_domain=...)`, a header LITH in zoned runs, `export_context`)
    re-resolves a `matrix_type` left unset, so a carbonate run does not
    keep the clastic sandstone.
  - `export_context` keeps the caller's explicit settings when
    `lithology_domain` or `lithology` is given.
  - A header VSHE does not switch `sw_model` to `north_sea`, in zoned runs
    as in `pp.interpret`.
  - Settings can differ by depth: `zones=` takes a list of intervals with
    their settings, or a table or CSV/Excel file (`top`, `bottom`, one column
    per setting, optional `name` and `depth_unit`). Each zone is interpreted
    with its settings and merged; `result.zone_results` holds the zone runs,
    each with the `zone_names` it computed.
    Each set of zone settings is run over the whole well and merged over the
    zone, so a zone changes only what its settings change (the GR baselines,
    irreducible water saturation and core model stay the whole well's).
    Overlapping zones (also across depth units), whole-run settings in a zone,
    repeated columns and invalid values, also against the whole-well
    settings, are refused before anything runs. Order: your zone values, your
    whole-well settings, LAS header zone values (`RW_Z1` ...), LAS header
    whole-well values, defaults; a header value yours override is named in
    the warnings. `pp.interpret`, `run_dynamic_workflow` and the batch
    processor apply every LAS header zone value (`M_Z1`, `GR_MIN_Z1`, ...,
    as well as Rw, Rmf and the shale point); the zoned runner takes the same
    path outside the
    "geolink" profile. The record of the settings you passed survives
    `dataclasses.replace`.
  - Zones can be your own segmentation of the well: a zone with only a name
    and depths changes nothing and names the interval, and a `well` column
    lets one table hold the zones of several wells. `result.zone_summary()`
    gives one row per zone with the columns of `formation_summary`, and
    `pp.export_context(..., layout="package", zones=...)` (CLI `--zones`)
    groups each well by its zones in place of its formations, with
    `zone_info` and `zone_name`. Each formation or zone is split into
    lithology groups, or is one group with `split_by_lithology=False` (CLI
    `--no-lithology-split`).
  - The workflow trace leaves out the library's internal config records.
  - Archie `a`, `m`, `n` and the net-pay cutoffs also take one value per
    lithology, e.g. `m={"sandstone": 1.8, "carbonate": 2.3}` or
    `sw_cutoff={"clastic": 0.5}`: each sample takes its lithology's (Archie)
    or family's (cutoffs) value, and the lithologies left out keep their
    defaults: the LAS header's value where it gives one (`M`, or `M_Z1`
    inside its zone), else the lithology default; `"*"` sets the value for
    every lithology not listed. Saturation routing follows a per-family clay
    cutoff. Unknown
    lithologies, two values for one cutoff family and out-of-range values are
    refused; not available under the "geolink" profile.
  - The COSTA brine and illite defaults apply only in the carbonate domain;
    MCP does not add the COSTA calibration to a clastic run of a COSTA well.
  - The porosity trace reports a set `rho_ma`/`dt_ma`/`nphi_ma` and the
    endpoints actually used.
- Checks and fallbacks of the MCP server, the modules and the CLI:
  - MCP checks `lithology_codes` paths against the allowed folders like the
    other file settings.
  - MCP `run_workflow` returns table steps (`formation_summary`) as `tables`;
    `workflow_step_kinds` gives requires/produces for `formation_water` and
    `mud_filtrate`; `list_sample_wells` describes every collection and lists
    the MDT pressure files apart from the wells.
  - `RwModule`/`calculate_rw` in "auto" use the header RW (at its RWT) and
    header RMF; unknown methods raise.
  - `create_pickett_plot` falls back to RDEP when there is no RT.
  - `total_porosity` rejects unknown methods.
  - `StandardWorkflow`: unset cutoffs follow `lithology` (carbonate 0.3/0.045/
    0.7), failed steps appear in `result.errors`, and an unsupported
    `saturation_model` is refused with a pointer to `pp.interpret`.
  - `petrophys-context` given a file says a folder is needed.
- Docs: the precedence of settings, header values and defaults is stated once
  for every route (settings reference, workflows guide); the header-parameter
  table lists every parameter read, with zone suffixes; reader override
  arguments and `MDTPickingConfig` are documented; lesson 10 shows a header
  value against a setting and a real zone override. The porosity lesson
  shows the hydrocarbon correction skipped for want of a measured Sxo.
- Wording: messages, docs and tutorials say "lithology" for the general
  case and name GeoLink only for the GeoLink data, profile and packages. The
  built-in lithology code table (Kingdom nomenclature) is
  `LITHOLOGY_CODE_LABELS` (`GEOLINK_LITHOLOGY_CODE_LABELS` is an
  alias); a run without lithology input records "No lithology curve or
  lithology intervals were available"; a `LITHOLOGY_GEOLINK` curve is
  reported as `vendor_lithology_log[LITHOLOGY_GEOLINK]` like any other code
  curve (benchmark reproduction keeps its packages' wording).
- `pp.plot(..., curves=[...])` plots only the named curves, each in its usual
  track; a curve no track takes gets a generic track. Names may be the file's
  own (`GRD`) or standard names (`GR`, `RDEP`, `NPHI`), found through the
  curve-name mapping and the alias tables; a crossover fill is kept only when
  its partner curve is selected too. `tracks=` and `curves=` together are
  refused with guidance; an empty or unknown selection says what the well has.
- Plotting checks: `curve_styles` rejects an unknown
  fill or colour before drawing and points `scale` to the track; a depth
  window outside the logged interval warns, a window of no height is
  refused; choosing tracks gives no warning about the curves left out.
- The automatic layout draws NMR porosity, free fluid and bound water
  (TCMR/CMRP_3MS, CMFF, BVI; converted from porosity units to v/v for the
  track) and NMR permeability (KSDR, KTIM), a second caliper (HD2, C2),
  uranium-free gamma ray (CGR), a SPECTRAL GR track (K, Th, U), the invasion
  inversion's Rt with the resistivities and a PRESSURE track (overburden,
  hydrostatic and Eaton pore pressure on one scale). TVD/TVDSS and the
  inversion's bounds and misfit are not drawn as generic curves.
- Curve-name mappings for mnemonics no alias table knows:
  `pp.read`, `pp.interpret` and `pp.plot` take `curve_aliases=` (a dict, or a
  CSV/Excel table of curve and standard name), stored on the well so reading,
  interpretation and plotting all use it; a mapped curve outranks other
  candidates; a mapping to a known alias (`RT`) stands for its standard
  curve (`RDEP`). `petrophys-context` takes `--curve-aliases FILE`. An
  entry that names no curve of the well, or is written the other way round
  (`{"GR": "GAMMA"}`), gives a warning with the likely correction.
- `create_histogram_comparison` takes `value_range` and `histtype="step"`;
  the multi-well overlay labels its depth unit and draws legible legend
  lines. The NMR panel titles its first track "GR / Tension" (CDF and TENS
  are cable tension) and states a constant T2 cutoff in the Swirr title.
- DCAL is read as a caliper only when its values are a hole diameter (median
  of 4 in or more); a differential caliper (caliper minus bit size) under that
  name is not used as CALI.
- Plots recognize the interpreted-curve names of other packages (Techlog,
  Geolog, IP, ELAN: VWCL, PIGN, SWE, SUWI, KLOGH, KINT, PAYFLAG, NETRES, ...);
  flag names such as PAY or RES are taken only for 0/1 curves. `LITH` is
  read as a lithology-code curve name.
- Under the benchmark profiles the saturation step names the equations its
  samples are routed to, with their counts.

- The published GeoLink, COSTA and Danish context packages are computed by
  context profiles, `geolink_v1`, `costa_v1` and `danish_v1` (aliases
  `geolink`, `costa`, `danish`), each running its package's own pipeline from
  the delivery's raw LAS files with the calculations the package used:
  `export_context(..., context_profile=...)`, `petrophys-context --profile ...`,
  and `petrophys.reporting.context_profiles.export_context_profile`. Without a
  profile, the current workflow runs. A profile fixes its package's settings
  and rejects `workflow_config` and the other interpretation options.
- The GeoLink profile reads the raw `eval` LAS, the NPD tops workbook and the
  GeoLink lithology codebook. The COSTA profile needs no files besides
  petrophys: the COSTA LAS, the per-well interpretation parameters and the
  reference calcite/dolomite volumes are bundled (`datasets/costa`,
  `_internal/data`). The Danish profile reads the raw LAS, the Danish tops
  workbook and the GeoLink lithology codebook.
- `scripts/generate_benchmark_context.py` regenerates a package. It accepts
  one LAS file or a folder from any eval collection and recognizes GeoLink,
  COSTA or Danish wells by name (`--dataset` overrides). With `--from-raw` it
  starts from the raw LAS files and writes both outputs: the interpreted
  ("processed") LAS files under `<output-dir>/processed_las`, then the
  context built from those. `--calculation-profile` (`geolink`, the default,
  or `current`) applies to both steps, and `--well` limits the run to the
  wells named. With the package's NPD tops workbook and lithology codebook
  next to the raw LAS folder (or given with `--tops-file` and
  `--lithology-mapping-file`), one command regenerates the whole GeoLink
  package:

  ```bash
  python scripts/generate_benchmark_context.py --from-raw --input-dir PATH/eval --output-dir NEW_DIR
  ```

- The category context exporter takes `group_layout`: per well, the
  formation/lithology groups to export instead of splitting by tops and
  lithology codes (the Danish profile's one group per formation).
- Under `"geolink"` the interpreted LAS keep the source's depth unit label
  (e.g. `FT`) and equivalent unit labels (`gAPI`, `m3/m3`), write the
  temperature parameters without units, and pad header columns by UTF-8
  bytes, as the eval package's files were written; `export_las` takes
  `depth_unit_label` and `pad_header_by_bytes` for this.
- Context packages use the eval package's layout: input curves are
  classified against the raw LAS files (`raw_input_dir`, default the sibling
  of a `*_processed` folder), value ranges use the eval package's rules (RMF
  from 0.01 ohm.m; no SP, SO or NMR rules), BS gets statistics and SO does
  not. Their notes describe the calculations of the profile used, and under
  `"current"` their depths carry the well's depth unit.

### Analysis tools

Functions for analyses `pp.interpret` does not run; see the Analysis
tools page.

- Th/K clay typing: `classify_clay_th_k` and `clay_typing_from_well` read the
  thorium/potassium ratio of spectral gamma ray in chart bands (feldspar,
  glauconite, mica, illite, mixed-layer, kaolinite, heavy-thorium minerals;
  bands configurable), with a potassium floor and an optional VCL cutoff.
- Step-profile invasion inversion: `invert_step_profile` solves Rt, Rxo and
  invasion diameter from three resistivities (or two and a measured Rxo),
  any mix of laterolog and induction readings, with tool responses J(di)
  given as `ToolResponse` tables; it reports the range of Rt and di that fits
  the readings within their noise, and the deepest reading's flushed-zone
  share. `forward_step_profile` models readings.
- Empirical saturation-height functions: `fit_saturation_height` and
  `fit_saturation_height_from_well` fit power-law, Cuddy (FOIL) and
  Skelt-Harrison functions to log Sw against vertical height above a known
  FWL; a TVD curve is required unless the well is declared vertical.
- Monte Carlo uncertainty: `monte_carlo` draws stated input distributions
  (normal, lognormal, uniform, triangular, with truncation), runs any
  vectorised calculation over scalars or curves, and reports percentiles and
  rank-correlation sensitivities; `correlated_depths` applies one error to a
  whole curve.
- Permeability heterogeneity: `lorenz_coefficient` and `dykstra_parsons`,
  thickness-weighted.
- Change-point zonation: `change_point_zonation` and `zone_well` (exact PELT on
  one or more curves, minimum zone thickness, penalty or `max_zones`).
- Thomas-Stieber shale distribution: `thomas_stieber` (laminar, dispersed and
  structural shale, sand-bed porosity and net-to-gross) and
  `laminated_sand_resistivity`.
- Raymer-Hunt-Gardner sonic porosity: `raymer_hunt_gardner` (exact and field
  forms) and `sonic_porosity(transform="raymer")`; Wyllie stays the default.
- Public `apparent_water_resistivity` (Rwa) and `sw_from_rwa`; apparent
  matrix density, slowness and volumetric PE (`apparent_matrix_density`,
  `apparent_matrix_slowness`, `apparent_matrix_volumetric_pe`,
  `apparent_total_porosity`); `calculate_winland_r35` and
  `classify_pore_throat`.
- `zone_statistics`: statistics and percentiles of any curves over any zones
  (tops file, table or intervals).
- Getting outside data in, for the tools that need more than logs:
  - `read_core` reads core analysis tables (CSV/Excel), finding columns by
    name (`Depth (m)`, `Por (%)`, `Kh Kl. (mD)`, `KAIR`, ...), converting
    percent porosity (also when labelled as a fraction), darcy and kg/m3, and
    core depths to the well's unit; unused numeric columns are reported.
    `CoreData.from_csv` warns when none of its columns is found.
  - `read_survey`, `minimum_curvature` and `attach_survey` (also
    `pp.read(path, survey=...)`, or `survey="vertical"`) add `TVD` and, with
    the depth reference's elevation, `TVDSS`; surveys come from CSV, Excel or
    LAS, with units from headers such as `MD (ft)` and azimuth optional.
  - `depth_reference_elevation` reads the elevation from EKB/EDF/EGL by LMF,
    or APD + EPD; `pp.read(path, survey=..., elevation=...)` sets it in place
    of the header's.
  - `read_tool_responses` reads digitized J(di) charts (long or wide, inches
    or cm, fraction or percent); `invert_well_invasion` inverts a well's
    curves and adds `RT_INV`, `RXO_INV`, `DI_INV` and the Rt range.
  - `fit_saturation_height_from_well` takes the FWL result of
    `fit_fwl`/`infer_contacts` directly and uses the matching TVDSS curve.
  - `eaton_from_well` runs Eaton on a well (air gap from the header: offshore
    the reference elevation, onshore EKB - EGL), and
    `calibrate_eaton_exponent` / `PorePressureResult.calibrate` fit the
    exponent to measured pressures.
  - `read_uncertainty_table` reads input distributions from Excel or CSV for
    `monte_carlo`.
- `read_formation_pressure` and `read_formations` accept headers with units
  in brackets (`TVDSS (m)`, `Formation pressure (bar)`, `MD (ft)`); the unit
  becomes the depth or pressure unit. Spectral gamma recognises a potassium
  curve in v/v or frac under a generic name such as `K`.
- Checks and messages: Winland R35 refuses percent
  porosity; Eaton refuses an offshore air gap of an onshore header and
  explains calibration points outside the logged interval; a poor normal
  compaction trend or saturation-height fit warns; Monte Carlo explains a
  function that is not vectorised; results print as short summaries.
- Pore pressure: `overburden_pressure`, `hydrostatic_pressure`,
  `fit_normal_compaction_trend`, `eaton_pore_pressure` (sonic and
  resistivity) and `pressure_gradient`, in m/ft and bar/psi/kPa/MPa.

### Interpretation

The default profile, `"current"`, computes the interpretation described
below; `"geolink"` keeps the GeoLink package's calculation in each case. A
bullet that names a library function describes that function for every
caller; under `"geolink"` the workflow keeps the package's calculation
wherever the package used the function.

#### Workflow defaults

- GR endpoints are picked from the log (`gr_min`/`gr_max` None: P5/P95), with
  a sandstone matrix in the clastic domain; with another `lithology_domain`
  the matrix is resolved within that domain. `standardize_input`,
  `vcl_cutoff` and `sw_cutoff` default to None and take their profile's
  value: input standardization is on under `"current"` and off under
  `"geolink"`, and unset net-pay cutoffs follow each sample's lithology
  family under `"current"` and are VCL < 0.3 and SW < 0.7 under `"geolink"`.
- LAS header parameters override settings left at their defaults (for
  example `RHO_FL` over an unset `rho_fl`); explicitly set values win.
- Without a measurement temperature, a given Rw is taken as the value at the
  median formation temperature and corrected along the temperature profile
  from there. The clean-wet Rwa fallback takes the P5 of the clean samples'
  apparent Rw at one reference temperature (the minimum-Rwa method), leaving
  out values saltier than saturated brine (260,000 ppm). When no Rw source is
  found,
  0.05 ohm.m is used with a warning (`allow_default_rw=True` by default; set
  it to False to skip saturation instead).
- PHIT and PHIE are always calculated by the run.
- Net-pay cutoffs are strict (`VCL < cutoff`, `PHIE > cutoff`,
  `SW < cutoff`).
- The reservoir flag that picks the SWIRR_APP samples before permeability is
  computed from the run's own cutoffs. A `RESFLAG` or `NET_RES` curve already
  on the well, as an interpreted LAS carries, is not used for it, and a
  warning says so; set `reuse_input_resflag=True` to pick Swirr from the
  supplied flag.
- `DynamicWorkflowConfig` validates every setting when it is created and
  raises `ValueError` listing all invalid fields with the accepted values.
- `lithology_domain` distinguishes the clastic and carbonate mineral routes
  from per-depth lithology labels. It defaults to `"clastic"` (set
  `"carbonate"` for carbonates, or `"auto"` to resolve it from the data);
  `matrix_type` defaults to `"sandstone"` in the clastic domain and to
  `"auto"` within any other domain, and an `"auto"` matrix stays inside the
  domain's family. When the matrix evidence points elsewhere, the domain wins
  and a warning names the conflict.
- Tabular readers (`io.read_csv`, `read_excel`, `read_csv_wells`,
  `read_ascii` without a units row, `Well.from_dataframe`) require
  `depth_unit` unless the depth header gives the unit (`DEPTH (m)`).
  `Well.depth_unit` accepts only metre or foot labels.
- `pp.read` reads Excel logs (`.xlsx`, `.xlsm`, `.xls`; `sheet_name=`) as it
  reads a CSV of the same table, and detects a semicolon or tab separator
  and a decimal comma. `Well.from_dataframe` reads `NAME (unit)` headers.
- Reading several wells with one tops or lithology table matches each well
  on its WELL header, else its file name; a well the table has no rows for
  is read without them, with one warning naming those wells. A tops table
  in the other depth unit, with no unit declared, raises ValueError when
  the other unit fits every row.
- `pp.read` messages say where a missing file was looked for and which
  names there are close, and name stray quotation marks, a path without an
  extension and a Windows path whose backslashes became control characters.
  A folder holding LAS, CSV and Excel files reads with one `depth_unit`.
- Settings are checked when the configuration is built: a value of the
  wrong type, a lithology file that does not exist, `peer_wells` that are
  not wells, a `lithology_curve` that does not hold lithology codes and
  values outside physical ranges (`rho_cl`, `swirr=0`, `rw` above
  100 ohm.m, sonic end points, `bht_f`) raise ValueError. A total depth
  above the deepest log sample is not used for the temperature gradient. A
  setting that has no effect in a run (peer core without its conditions,
  `cec` without `b_coeff`, `rw_meas_temp_f` without an Rw, curve mappings
  with standardization off) gives a warning saying why.

#### Inputs

- A LAS file whose index curve and STRT/STOP/STEP give different depth units
  (m and ft) gets a warning naming the unit used.

- Input curves are standardized by default: vendor mnemonics get canonical
  aliases, units such as kg/m3 or percent porosity are converted, and
  implausible values are removed with a `DataQualityWarning`. The screening
  also removes compressional slowness below 38.5 us/ft, faster than any
  rock matrix, and SP beyond +/-500 mV.
- A resistivity curve whose maximum of 1000 ohm.m or more repeats in three or
  more samples, such as a tool saturated at 2000 ohm.m or a 1000/conductivity
  resistivity capped at 100000 ohm.m where the conductivity reads zero, has
  those ceiling samples removed, with a `DataQualityWarning`. The
  normalization step records them under `ceiling_screening`.
- Further vendor mnemonics are recognized: the density and microlog calipers
  CALD, DCAL and MCAL (CALI), the density correction CORR (DRHO), the medium
  induction RILM (RMED) and the shallow laterolog LL3/RLL3 (RSHA).
- Every workflow step records every curve it created, changed or removed:
  the curves are compared before and after each step, and any it wrote
  without listing (such as PRIME_OIL_TARGET from the net-pay step's
  hydrocarbon analysis, TEMP_LOG from the temperature step, or BADHOLEFLAG)
  join its `output_curves`; removed curves are in `curves_removed`. The
  normalization step lists the curves it created or changed rather than
  every curve of the well. Under the benchmark profiles `output_curves`
  keeps the eval package's list, which its context records repeat, and the
  further curves are in the step's `also_written`.
- Lithology interval names are read for the rock they describe
  (`petrophys.interpretation.lithology.labels.lithology_group_for_label`):
  abbreviations (Sst, SH, LST, DOL, ANH), modified names (shaly sandstone,
  argillaceous limestone, "Sandstone, calcareous"), carbonate textures, coal
  and igneous rocks. Mixed names stay without a group, as unrecognized
  names do.
- `DynamicWorkflowConfig.lithology_codes` takes the codebook of a lithology
  code curve whose codes are not those of the built-in lithology code table: a CSV or Excel
  codebook, a DataFrame or a code-to-name mapping. Its rock names give each
  code's lithology group, the lithology domain and the lith_vcl permeability
  transform; a code curve under another lithology mnemonic (FACIES,
  LITHOLOGY, ...) is found by the codebook's codes, and a codebook that no
  curve holds is named in the warnings.
- `read_lithology_codes` recognizes more codebook headers (rock type, rock
  name, facies, lith code, id, label, ...) and reads a table of two other
  columns as code, then rock name.
- A BHT at a total depth at or above the surface (a well logged from 0 only)
  does not fail the temperature step: the default gradient is used, with a
  warning.
- A logged vendor density porosity (DPOR, DPHI, DPHZ, PORD) or sonic porosity
  (SPOR, SPHI, PORS) gives PHID or PHIS wherever its value, in v/v or
  converted from percent, lies within 0-0.5; the computed porosity fills the
  other samples. The porosity step records how many samples each came from.
- A LAS header without BS takes its bit sizes from numbered bit runs (BS1,
  BS2, ...), each over the depths it drilled (BSD1, BSD2, ...); bad hole
  compares the caliper with the bit of each depth.
- TEMP and TMPC curves are read in the unit their label gives, and
  `standardize_curves` converts a degC or K temperature curve to F and
  screens it on its own scale.
- A header bit size outside 3-36 in, such as a -999.25 null or a value in
  millimetres, is ignored, with a warning unless a BS curve covers every
  sample.
- A header CEC labelled meq/g is converted to meq/100 g.
- Values the run has to assume are reported in `result.warnings` and in the
  `~Other` section of `result.to_las()`: the 30 C/km temperature gradient
  when there is no BHT, an Rw without its temperature taken at the median
  formation temperature, an Rw estimated as P5 of the clean-sample Rwa, and
  Swirr assumed at 0.20. No bit size is assumed; a warning names the depths
  where a caliper reading has no bit size (see "Bad hole").
- The workflow finds the lithology code curve by one rule: the curve named
  by the setting `lithology_curve`, else `LITHOLOGY_GEOLINK`, else, with
  `lithology_source` `"auto"` or `"vendor"`, the first curve under a
  lithology-code name (`LITHOLOGY`, `LITHOFACIES`, `FACIES`, `LITH_CODE`,
  `ROCK_TYPE`, ...) whose values pass the code-log checks: 20 or more finite
  samples, whole numbers from 0 to 99 with at most 50 distinct values, half
  or more of them codes of the built-in table. That one curve gives the vendor lithology,
  an `"auto"` matrix and the per-depth lithology groups behind the net-pay
  cutoffs and exclusions. The run records it on the well, and context
  exports, the diagnostics, the zoned batch runner, `steps.net_pay`,
  `PermeabilityModule`, `core_data_from_well`, `IntervalAnalyzer` and the
  log viewer read the same curve. A curve other than `LITHOLOGY_GEOLINK` is
  named in a warning saying that its codes are decoded with the
  built-in lithology code table; the trace gives the lithology source as
  `vendor_lithology_log[MNEMONIC]`, and its `input_context` step reports the
  curve as `vendor_curve`. Names that only contain LITH or FACIES, such as
  `LITHO_PATT` in the Daydream 1 logs, are not read as lithology codes, and
  curves the workflow writes (`LITHO`, `LITHO_CLASS`, `MMQC`, `GT_*`) never
  are.
- `standardize_curves` detects percent core saturations under a fraction or
  blank label and converts them to v/v with a warning. When two curves map
  to one standard mnemonic, the curve with more valid samples takes the
  slot and the other keeps its own mnemonic.

#### Bad hole

- No bit size is assumed. The caliper washout test (CALI - BS > 1.5 in) and
  the large-hole test (BS > 17.5 in) run only where a bit size is known,
  sample by sample: the BS curve, else a 3-36 in header bit size, which also
  fills the gaps of a BS curve. A |DRHO| above 0.15 g/cm3 flags washout
  wherever DRHO is logged, also in a well without a caliper, so where no bit
  size is known BAD_HOLE comes from |DRHO| alone. A warning names the depths
  where a caliper reading has no bit size, and a well with a caliper but
  neither a bit size nor DRHO skips the step and says why. The trace counts
  the samples without a bit size (`no_bit_size_samples`), those given the
  caliper test (`caliper_test_samples`) and those neither test reaches
  (`not_assessed_samples`; BAD_HOLE is 0 there), and its `bit_size_source`
  is `"BS_curve_and_well_header"` where the header fills gaps in the BS
  curve and `"none"`, with `bit_size_used` None, in a well without a bit
  size. The "geolink" profile keeps the package's 8.5 in bit size where a
  well has no BS curve or header bit size.

#### Clay volume

- VCL is the minimum of VCL_GR and VCL_ND. VCL_SP from baselines picked over
  the whole log (P5 and P95) is reported but left out of the minimum, and
  fills only depths with no GR or neutron-density value: such baselines
  mostly measure the drift of the SP shale baseline. Setting both `sp_min`
  and `sp_max` puts VCL_SP into the minimum.
- VCL_ND is measured from the clean line of the run's matrix, sandstone by
  default. An unresolved `matrix_type="auto"` uses its domain's matrix or
  sandstone, and the run warns when the matrix later resolves to one with
  another clean line.
- The VCL_ND clay point is picked in the well's good-hole shales: the median
  RHOB and NPHI of at least 20 shale-lithology samples, else of the samples
  with VCL_GR above 0.9, else (2.62 g/cm3, 0.38). A point less than 0.10 v/v
  of neutron above the clean line leaves VCL_ND out of VCL, with a warning.
  The setting `nd_clay` sets the point instead; zoned runs use the
  whole-well point.
- VCL_ND is undefined where NPHI lies above both the clay point and the
  clean line's 40 p.u. point, outside the neutron-density triangle; VCL
  comes from the other indicators there.
- `sp_min` is the shale SP baseline and `sp_max` the clean-sand line. User
  baselines are checked against the SP polarity read from the SP-GR rank
  correlation (`sp_polarity`): an order that contradicts it is warned about,
  and equal values are rejected.

#### Porosity and core correction

- Evaporites take their apparent density-log reading as the density-porosity
  endpoint: halite 2.032, anhydrite 2.977 and gypsum 2.351 g/cm3
  (Schlumberger, 1972). The potash salts have endpoints of their own (the
  minerals sylvite, carnallite, polyhalite and kainite), which lithology codes
  27 and 32-34 and the labels "Sylvite" and "Potash" map to.
- Halite is also recognised from its logs where no lithology code calls a
  sample an evaporite: RHOB 1.95-2.15 g/cm3, GR below 15 API, deep
  resistivity above 500 ohm.m (without one, a caliper 0.5 in under the bit
  size) and, where logged, NPHI up to 0.05 and PEF 4.0-5.5. Such samples
  take the halite endpoints and are left out of net reservoir and net pay.
- Clay porosity (`phi_cl` in PHIE = PHIT - VCL x phi_cl) comes from the
  shale samples recomputed on the reservoir matrix endpoints, the basis of
  the PHIT it is subtracted from. Dual Water uses the same value.
- A LAS header PNSH, the neutron log's reading in shale, gives the clay
  porosity together with RBSH (shale density) as a neutron-density shale
  point. PNSH alone is set aside, as it is not a clay porosity.
- GeoLink code 10 ("Porous Limestone") takes limestone endpoints and code 26
  ("Gap", no lithology recorded) no lithology group; context exports name
  them so.
- A context package of a well without input lithology (no lithology-code
  log and no lithology intervals) groups each formation by the lithology the
  interpretation computed (LITHO), bed by bed; beds of the minimum group
  thickness or less are left out, as for input lithology. The context records
  the lithology as computed from the logs, not as ground truth, and reports
  its distribution rather than LITHO statistics.
- In neutron-density crossover (PHID > PHIN + 0.05), PHIT is the
  root-mean-square of PHID and PHIN (Gaymard and Poupon, 1968) only with
  `hc_type="gas"`; in oil wells it is the density porosity.
- Coal takes the chartbook slowness of 120 us/ft (bituminous coal reads
  110-140 us/ft) and the gypsum group the table's 52.5 us/ft.
- The same-well core porosity correction is fitted to PHIT, since the helium
  porosity of cleaned, dried plugs is total porosity (API RP 40, 1998), and
  PHIE_CC = PHIT_CC - (PHIT - PHIE), kept within 0 and PHIT_CC. A fit is
  used only with at least 8 core-log pairs, a positive slope and an r^2 of
  at least 0.5, for the whole well or a formation; a rejected fit is listed
  with its reason in the trace (`core_rejected_models`).
- Input `PHIE_CC` and `PERM_CC` curves are kept as `PHIE_CC_INPUT` and
  `PERM_CC_INPUT` while the run writes them from its own core model.
- Library functions: `MineralProperties.log_density` and
  `MatrixProperties.get_log_density()` give the evaporite endpoints, which
  `PorosityModule` and `steps.porosity` use. `total_porosity` takes
  `gas_method` (`"rms"`, the default, or `"density"`) and counts a negative
  neutron as 0 in the root-mean-square; `PorosityModule` follows its
  default, and `steps.porosity` takes `hc_type` (`"oil"`, the default, keeps
  density porosity in crossover). `sonic_porosity` applies the compaction
  factor before the 0-0.5 limits (Hilchie, 1978). Oil and heavy oil have a
  hydrogen index of 1.0, so the opt-in hydrocarbon correction leaves NPHI
  unchanged for oil. The mineral table's clay U is 9.81 b/cm3 (3.7 x 2.65).
  `calculate_elastic_properties` returns floats for scalar inputs, and
  `ArchieParameterDefaults` reads "porous_limestone" as limestone and "gap"
  as generic clean rock.

#### Water and mud

- The header CLFL is read as mud-filtrate chlorides in mg/l, the mud-report
  unit (ppm, kppm and % labels are converted), into the setting
  `rmf_chlorides_mg_l`. Unless an Rmf or `rmf_salinity_ppm` is given, Rmf
  comes from the NaCl equivalent, Cl x 58.44/35.45, turned into ppm with the
  brine density of McCain (1991); the trace shows each step.
- The mud-sample temperatures MFST and MST stand in for RMFT and RMT as the
  measurement temperatures of Rmf and Rm.
- A header RW without RWT is taken at the header formation temperature FMT
  when there is one. The header OBM flag sets `mud_type`, WTSAL the water
  salinity (not used when the header gives RW) and FLSAL the mud-filtrate
  salinity; the "geolink" profile reads none of them.
- `gradient_f_per_100ft` sets a geothermal gradient, defaulting to the header
  TGRAD (degC/100 m, degC/km or degF/100 ft). A BHT anchors the profile at TD
  and is used ahead of a header gradient; a gradient you set is used instead
  of the header BHT. `rw_meas_temp_f`, `rmf_meas_temp_f` and `rm_meas_temp_f`
  must be 0-700 degF.
- With `mud_type="OBM"` no SXO or MHI is computed, and Rw is not estimated
  from the SP or from Rxo/Rt: they need a conductive, water-based filtrate,
  and the 200 ohm.m Rmf written for oil-based mud is a placeholder. Rw then
  comes from `rw`, the header or the clean-wet Rwa estimate.
- Temperature helpers accept any recognised unit label, such as "degF",
  "deg C" or "K", and raise `ValueError` on an unknown one:
  `arps_equation`, `correct_resistivity_temperature`,
  `calculate_temperature_gradient`, `TemperatureProfile` and its `from_bht`,
  `from_gradient_c_per_km` and `from_two_points`, `rw_from_salinity`,
  `salinity_from_rw`, `rw_from_ssp` and `estimate_nacl_resistivity`.
  `estimate_gradient_from_bht` and `rw_at_reference_temperature` convert
  their default reference temperatures (68 and 75 degF) to the caller's
  unit, `RwModule` and `RmfModule` read TEMP and TMPC by their unit, and
  `convert_units` and `convert_curve_units` convert between temperature
  scales. Celsius labels such as "deg C (from log)" are converted wherever
  header temperatures are read.

#### Invasion

- The workflow's INVASION_RATIO reads, sample by sample, RXO (the
  flushed-zone microresistivity reading), else the shallow RSHA and then
  RESS, else the RMED/RESM medium/deep proxy; the trace records the order
  (`rxo_curve_order`) and a decision naming the samples each curve gave.
- `InvasionModule` reads the flushed-zone RXO first: its default
  `rxo_curve` is `"RXO"`, and after `rxo_curve` it tries RXO, RSHA and RESS
  in that order; `rxo_curve="RSHA"` reads RSHA, RXO, RESS.

#### Saturation

- The salinity of the Rw the workflow uses (from `salinity_ppm`, or the NaCl
  equivalent of an Rw given with its temperature or read from the header)
  chooses between the shaly-sand models: 30/6-5 with Rw 0.07 ohm.m at
  230 degF routes to Simandoux. A `salinity_ppm` set next to an `rw` of
  another water is not used, with a warning. A setting given by the caller
  (`rmf_chlorides_mg_l`, ...) is used even when it equals the header value.
- The saturation exponent `n` may differ by lithology (`n={"sandstone": 1.8,
  "shale": 2.2}`) under every saturation model; `simandoux`,
  `modified_simandoux`, `waxman_smits` and `dual_water` take one `n` per
  sample.
- The Rw step compares the Rw in use with the apparent water resistivity of
  the clean porous rock (Rwa = Rt x phi^m / a; Asquith and Krygowski, 2004)
  and warns when the median Rwa is below half of it, which puts Archie Sw
  above 1.4 in more than half of that rock; the trace records
  `rwa_median_over_rw` and `rwa_p5_over_rw`.
- `sw_model="modified_simandoux"` and `modified_simandoux()` add the modified
  Simandoux equation, 1/Rt = phi^m Sw^n/(a Rw (1 - Vcl)) + Vcl Sw/Rcl, the
  form many service-company reports use. `simandoux()` is documented as the
  Bardon-Pied form, without the (1 - Vcl).
- `rho_hc` and `hc_hydrogen_index` set the hydrocarbon density and hydrogen
  index of the neutron-density hydrocarbon correction; `rho_hc` also sets the
  hydrocarbon density of the core matrix-density check of auto lithology.
  Unset, they keep 0.8 g/cm3 and 1.0 for oil and 0.2 g/cm3 and 0.3 for gas
  (0.25 g/cm3 for gas in the matrix check). `correct_density_hc`,
  `correct_neutron_hc` and `HcCorrectionModule` take them as `rho_hc` and
  `hydrogen_index`.
- A saturation run whose Sw is NaN at every depth fails, naming the likely
  cause (no valid Rw, deep resistivity or porosity).
- Archie a, m or n set outside 0.3-3, 1-4 and 1-5 is used with a warning, and
  porosity from the neutron log alone (no density) gets a warning.

- The Indonesia equation is the published one of Poupon and Leveaux (1971),
  1/sqrt(Rt) = [Vcl^(1 - Vcl/2)/sqrt(Rcl) + phi^(m/2)/sqrt(a Rw)] Sw^(n/2),
  in `petrophys.interpret` and `steps.saturation`. The "geolink" profile
  keeps the package's variant.
- Dual Water runs only with a supplied bound-water resistivity: `rwb`, a
  header RWB or `steps.saturation(rwb=...)`. Rwb is not estimated. With
  `sw_model="auto"`, shaly carbonate samples without it take the clastic
  routing (Waxman-Smits with CEC and B, Simandoux at a salinity of
  20,000 ppm or more, otherwise Indonesia), and the trace counts them;
  `sw_model="dual_water"` without Rwb stops the saturation step with an
  error that asks for it. The equation is the dual-water model of Clavier,
  Coates and Dumanoir (1984), with the bound-water saturation capped at
  0.999.
- Waxman-Smits (Waxman and Smits, 1968) takes `cec` in meq/100 g and
  converts it to the meq/g of the equation. A `cec` outside 0-150 meq/100 g
  (pure smectite is 80-150) is rejected, and one below 1 is warned about as
  a likely meq/g value. `sw_model="waxman_smits"` without `cec` and
  `b_coeff` gives a clear error.
- The Waxman-Smits saturation, solved on total porosity, is reported on the
  effective basis, SW = 1 - (PHIT/PHIE)(1 - Swt), so BVW, HCPT and net pay
  pair it with PHIE like every other model. Where SW is paired with PHIE_CC,
  Waxman-Smits and Dual Water take PHIE_CC + (PHIT - PHIE) as total
  porosity. Samples whose Swt lies below the bound-water fraction
  (PHIT - PHIE)/PHIT get SW = 0 and are counted in the trace.
- The setting `sw_routing_vcl_cutoff` is the clay volume above which
  `sw_model="auto"` uses a shaly-sand equation. Unset, it is `vcl_cutoff`,
  or 0.3 when that is unset too, so net rock with 0.3-0.4 clay (below the
  clastic standard of 0.40) gets the shaly-sand equations. The SWIRR_APP
  pick uses the same cutoff.
- Library functions: `SaturationModule` does not estimate Rwb
  (`rwb_fallback=None` by default; pass `rwb`, or `rwb_fallback=0.3` for
  Rwb = 0.3 Rw), and `effective_waxman_smits=True` reports the Waxman-Smits
  Sw on the effective basis. `SaturationResult` has `fallback_counts` (for
  example `{"dual_water->indonesia": 12}`) and
  `waxman_smits_clipped_to_zero`. `calculate_sw` warns with the number of
  fallback samples, and `StandardWorkflow` takes `WorkflowConfig.rwb`.

#### Mineral solvers

- Mineral solvers are versioned. `MultiMineralSolverV1`, the clastic
  workflow's solver, takes VCL and PHIE as fixed inputs, divides each
  equation row by its nominal one-sigma tool uncertainty before the
  non-negative least squares, the weighted solve of Mayer and Sibbit (1980)
  and Quirein et al. (1986): `row_uncertainty`, default `ROW_UNCERTAINTY`
  (closure 0.01 v/v, RHOB 0.02 g/cm3, NPHI 0.02 v/v, U 0.5 b/cm3, DT
  2 us/ft), and leaves unsolved the samples whose mineral total misses
  `1 - VCL - PHIE` by more than 0.05. `row_uncertainty=None` solves the rows
  unscaled, which the "geolink" profile uses. `MultiMineralSolver` is
  version 2 (also exported as `MultiMineralSolverV2`).
  `DynamicWorkflowConfig.mineral_solver_version` selects the clastic
  workflow solver (default `1`), and `LithologyModule(solver_version=...)`
  selects it there (default `2`).
- The carbonate domain takes calcite and dolomite from bulk density and
  closure in closed form over whole arrays (Clavier and Rust, 1976; module
  `petrophys.interpretation.lithology.carbonate_minerals` with
  `solve_calcite_dolomite` and `solve_carbonate_minerals`): a whole well
  takes well under a second. A sample whose bounded solution misses RHOB by
  more than 0.02 g/cm3 is left unsolved and flagged. The setting
  `carbonate_calibration` chooses the constants: `"generic"`, the default,
  takes clay from VCL at `rho_cl`, effective porosity, the pore fluid
  `rho_fl`, and calcite 2.71 and dolomite 2.87 g/cm3; `"costa"` is the COSTA
  field calibration (illite from GR, total porosity, a 1.1 g/cm3 brine and
  a 2.847 g/cm3 dolomite) and the "geolink" profile's default, which there
  keeps the per-sample solve of the COSTA notebooks.
- Mineral inversion reports non-resolvable and zero-volume failures as
  missing values with QC flags. A reduced calcite-dolomite model is
  available when carbonate evidence supports it. Workflow traces record
  mineral selection.
- `LithologyModule(solver_options=...)` passes options such as
  `row_uncertainty` to its solver.

#### Permeability, saturation height and NMR

- Special core analysis: `read_capillary_pressure` reads plug capillary
  pressure curves (Pc against Sw, with each plug's porosity and
  permeability) from a CSV or Excel table. Columns are found by name with any
  unit in brackets ignored, the pressure unit comes from the header or
  `pc_unit=`, and a mercury saturation becomes Sw = 1 - SHg; a saturation
  that rises with pressure is refused as not Sw. A sample column filled on
  each plug's first row only (merged cells) is carried down, and a pressure
  unit written as a suffix (`Pc_kPa`, `PC_BAR`) is read. `.thomeer()` fits the Thomeer
  (1960) hyperbola to each plug (entry pressure Pd, pore geometrical factor
  G, Bv-infinity), `.to_reservoir(fluids)` converts lab Pc by the ratio of
  sigma cos(theta), and `fit_j_function_from_core` fits one Leverett
  J-function to all plugs at reservoir conditions.
- Fluid contacts from capillary pressure: `estimate_fwl_from_logs` finds the
  free-water level at which a core saturation-height model best matches the
  log Sw, with a 95% range (or says the FWL lies deeper or shallower than the
  logged interval when the logs miss the transition zone, and the R2 of the
  match), and
  `water_contact_from_fwl` places the water contact above the FWL by the
  entry-pressure height. The search for the FWL starts at the shallowest
  sample used, and a 95% range cut by the edge of the search is flagged.
- The volumetric functions (`calculate_ooip`, `calculate_ogip`,
  `calculate_hcpv`) and the stand-alone net-pay functions read a porosity,
  saturation or cutoff above 1, up to 100, as percent, and refuse negative
  areas and thicknesses, values above 100 and non-positive formation volume
  factors.
- `cutoff_sensitivity(result)` starts from the cutoffs the run set, and
  names its base in `attrs["base_cutoffs"]`; a run on the standard
  per-lithology cutoffs is described in the docstring.
- `swirr` takes one value per lithology, as Archie a, m and n do; other
  lithologies keep the automatic estimate.
- Fraction settings (`swirr`, `vcl_cutoff`, `phie_cutoff`, `sw_cutoff`,
  `phi_cl`, `sw_routing_vcl_cutoff`) given above 1, up to 100, are read as
  percent: `swirr=15` is 0.15.
- `perm_method="wyllie_rose"` and `wyllie_rose()` follow the published
  Wyllie and Rose (1950) family k = P phi^Q / Swi^R (k in mD, phi and Swi as
  fractions), with presets in `WYLLIE_ROSE_PRESETS`: `morris_biggs_oil`
  (62,500, 6, 2; sqrt(k) = 250 phi^3/Swi, Morris and Biggs, 1967), the
  default; `morris_biggs_gas` (6,241, 6, 2); `tixier` (62,500, 6, 2; Tixier,
  1949); and `timur` (8,581, 4.4, 2; Timur, 1968). `p`, `q` and `r` override
  a preset.
- `lith_vcl` permeability covers clean rock with VCL = 0.
- `saturation_height_from_logs` converts heights in metres to feet for
  Pc = 0.433 (SGw - SGhc) h, takes a `tvd_curve`, and warns when it takes
  the depth index as true vertical depth; `fwl_depth` must be in the same
  unit and on the same datum.
- NMR: the clastic T2 cutoff default (`T2_CUTOFF_CLASTIC`) is 33 ms (Coates,
  Xiao and Prammer, 1999), and `bvi_from_bins` and `ffi_from_bins`, also as
  `NMRData` methods, split the bin that straddles the cutoff by its log-T2
  width (`t2_fraction_below_cutoff`). `partial_bins=False` keeps the
  whole-bin split; with `cutoff_ms=32.0` it matches the MBVI supplied with
  the Phillips data set.
- `calculate_hcpv_from_well` converts metric net pay to feet, weights
  porosity by thickness and SW by pore volume, and raises `ValueError`
  without NET_PAY, PHIE or SW.

#### Net pay

- Unset cutoffs take each sample's lithology-family standard from
  `STANDARD_CUTOFFS`: clastic VCL < 0.40, PHIE > 0.08, SW < 0.60; carbonate
  VCL < 0.30, PHIE > 0.045, SW < 0.70. Limestone, chalk, dolomite and marl
  groups are carbonate, every other group clastic, and samples without a
  group take the family of the run's lithology. A cutoff set in the config
  applies to every sample. The trace records the cutoffs used
  (`cutoffs_by_family`, `cutoff_source`).
- Evaporites (halite and the other salts, anhydrite and gypsum), coal,
  halite recognised from its logs and bad-hole samples (BAD_HOLE = 1) are
  not net reservoir or net pay; net rock keeps them. Evaporites and coal
  come from the per-depth lithology (a code log or intervals), never from
  `lithology_domain`. Where no bit size is known, BAD_HOLE comes from
  `|DRHO| > 0.15 g/cm3` alone (see "Bad hole"). The trace parameter
  `excluded_from_net_reservoir_and_pay` gives the samples and thickness
  removed per kind, and the MCP assumptions name them.
- SWIRR_APP is picked in clean, porous, reliable rock: VCL below the
  saturation-routing cutoff, PHIE above `phie_cutoff` (0.08 when unset, in
  every lithology), and none of the samples left out of net reservoir.
- `steps.net_pay` applies the same per-family cutoffs and exclusions
  (`exclude_non_reservoir=False` for the cutoffs alone). `NetPayModule`,
  `summarize_pay` and `calculate_net_pay` apply one cutoff set without
  exclusions, so summarize the NET_PAY of an interpreted well with
  `result.summary(top, bottom)`.
- Thicknesses are summed over each sample's midpoint cell from the actual
  depths, so mixed sampling rates add up and data gaps count for nothing:
  `NetPayModule` (gross is the sum of the cells), `summarize_pay`, the
  sensitivity tools, `calculate_net_gross`, the zone and formation
  summaries, `calculate_hcpv_from_well`, `flag_intervals` (which by default
  joins only samples whose cells touch) and, under `"current"`,
  `result.summary()`.

#### Classes and reservoir quality

- Each property has one class table, used wherever the property is
  classified: `petrophys.model.CLASS_TABLES` holds the classes of clay
  volume (VCL), porosity (PHI), permeability (PERM), water saturation (SW),
  BVW, RQI, FZI, MHI, Rxo/Rt (RXO_RT) and reservoir quality, each with the
  unit of its values, the label of a sample without a value (such as
  NO_VCL_DATA) and the source of its edges. The classification enums of
  `petrophys.model` and their helpers, the diagnostics analyzers and context
  exports classify through it, so a property gets the same class in every
  context category.
- Porosity and permeability grade clay volume as CLEAN_FORMATION
  (VCL < 0.25), MODERATE_CLAY and SHALY_FORMATION (VCL > 0.50). FZI takes
  the global hydraulic elements GHE_1 to GHE_10 of Corbett and Potter
  (2004), with BELOW_GHE_1 below 0.0938 um. BVW is graded from VERY_LOW_BVW
  (below 0.03) to HIGH_BVW (0.10 and above). A section that classifies one
  property gives a sample without a value its table's `NO_*_DATA` label.
  LOW_SW marks SW below 0.60, the WATER_BEARING edge; the Rxo/Rt classes of
  the permeability context (0.8 and 1.2) read the workflow's
  INVASION_RATIO; and `mhi_confirmation` in the permeability metadata counts
  the samples of each MHI class.
- Reservoir quality is graded from effective porosity, permeability and clay
  volume, with the limits of the sample's lithology family. Every grade is
  net reservoir rock of that family as net pay counts it
  (`STANDARD_CUTOFFS`), FAIR_RESERVOIR adds permeability above 0.1 mD, the
  tight limit, and only clean rock (CLEAN_FORMATION) is excellent:

  | Grade | Clastic | Carbonate |
  | --- | --- | --- |
  | EXCELLENT_RESERVOIR | PHIE > 0.15, PERM > 10 mD, VCL < 0.25 | PHIE > 0.15, PERM > 10 mD, VCL < 0.25 |
  | GOOD_RESERVOIR | PHIE > 0.10, PERM > 1 mD, VCL < 0.40 | PHIE > 0.10, PERM > 1 mD, VCL < 0.30 |
  | FAIR_RESERVOIR | PHIE > 0.08, PERM > 0.1 mD, VCL < 0.40 | PHIE > 0.045, PERM > 0.1 mD, VCL < 0.30 |

  A sample whose three values are valid but miss FAIR_RESERVOIR is
  NON_RESERVOIR, and one without PHIE, PERM or VCL is NO_QUALITY_DATA.
  Porosity is read from PHIE_CC, then PHIE, then PHIT, permeability from
  PERM_CC, then PERM, PERM_CORE and PERM_TIMUR, and clay volume from VCL,
  then VCLAY, then VSH, as net pay reads it. The family is the sample's
  net-pay family in the workflow and the family of its lithology in a
  context, where a compound name goes by the rock it ends with (SANDY
  LIMESTONE is carbonate, CALCAREOUS SHALE clastic); a sample without
  lithology, GeoLink code 26 ("Gap") included, takes the family of the run
  that interpreted it. PRIME_OIL_TARGET (POTENTIAL_PAY with
  EXCELLENT_RESERVOIR) and PRIME_GAS_TARGET (STRONG_GAS_INDICATION with
  GOOD_RESERVOIR or EXCELLENT_RESERVOIR) therefore depend on clay volume
  too.
- The diagnostics analyzers (`PorosityAnalysis`, `PermeabilityAnalysis`,
  `SaturationAnalysis`, `HydrocarbonAnalysis`, `LithologyAnalysis`) and
  `run_comprehensive_analysis` classify with `CLASS_TABLES` by default
  (`class_tables="current"`). With the default tables, the
  reservoir-quality sections of porosity, permeability, hydrocarbon and
  lithology need clay volume (VCL, VCLAY or VSH) as well as porosity and
  permeability, and apply clastic limits unless `lithology_family` is given.
- Library functions: `classify_porosity` and `classify_sw` read values as
  fractions unless `percent=True`, and warn when fractions look like percent
  (porosity above 1, SW above 2), so an unclipped SW of 1.02 is
  WATER_BEARING. `get_vcl_quality_code` gives 0 and `get_invasion_type` None
  for a value that is not finite; `is_clean_sand` and `is_shale` default to
  the VCL classes (`threshold=None`). `InvasionModule` counts its
  conductive, minimal and resistive percentages with the classes of its
  `class_bands`, `CLASS_TABLES["RXO_RT"]` by default.

#### Context exports

- A context export computes each curve's statistics once per lithology
  group, for all six category files, resolves lithology labels once per
  distinct value, and does not read the written files back.
- After a context export's value-range QC, PHI_NORM, RQI and FZI are
  recomputed from the porosity (PHIE_CC, then PHIE, then PHIT) and
  permeability (PERM_CC, then PERM) the QC left, and are empty wherever
  either input is missing, rejected samples included; PHI_Z takes the
  recomputed PHI_NORM, and BVW is empty where no porosity is left. The
  category exporter recomputes these logs for each group right after its
  QC.
- Depths and intervals carry the well's depth unit: `formation_info` and
  `lithology_group_info` give it as `depth_unit`, the bad-hole intervals of
  `BADHOLE_QC_CHECK` end with it, and the KH note gives KH in mD times the
  depth unit.
- The bad-hole QC of a context (`BADHOLE_QC_CHECK` and the BAD_HOLE note)
  follows the rule of the bad-hole step, so it agrees with the BAD_HOLE
  curve, and counts the samples without a bit size (`no_bit_size_points`)
  and those neither test reaches (`not_assessed_points`).

### Features

- `import petrophys as pp` holds the names most work needs: `read`,
  `interpret`, `plot`, `impute`, `export_context`, `read_formations`,
  `sample`, `list_samples`, `Well` and `DynamicWorkflowConfig`. Everything
  else is imported from its package (`petrophys.workflow`,
  `petrophys.model`, `petrophys.reporting`, ...).
- One name for each input across the front door: `formations=` (formation
  tops) and `lithology=` (lithology intervals) in `pp.read`, `pp.interpret`,
  `pp.plot` and `pp.export_context`, `lithology_codes=` (the codebook of a
  lithology code curve) in the last three, and `top=`/`bottom=` for a depth
  window. `pp.interpret(..., lithology=)` also takes the name of the well's
  lithology code curve or a lithology domain.
- `bad_hole` (`cali`, `drho`, `large_hole` thresholds), `perm_coefficients`
  (`a`, `b`, `c` of the Timur, Coates-Denoo or Wyllie-Rose equation) and
  `mineral_endpoints` (`rhob`, `nphi`, `pef`, `u`, `dt` per mineral or
  water of the clastic mineral solve) take the values you give over the
  defaults. The trace, the plots and the context notes use them. LAS
  header parameters give them one value each (`BADHOLE_CALI`,
  `BADHOLE_DRHO`, `BADHOLE_BS`; `PERM_A`, `PERM_B`, `PERM_C`;
  `QUARTZ_RHOB`, `WATER_NPHI` and the other `<MINERAL>_<PROPERTY>`), also
  per header zone (`BADHOLE_CALI_Z1`), below the values you pass.
- `pp.list_samples(kind)` lists one kind of bundled data: `"logs"`, `"nmr"`,
  `"pressure"`, `"core"`, `"geology"` or `"templates"`
  (`petrophys.datasets.sample_kind` names a sample's kind), as well as one
  collection such as `"costa"`.
- `pp.plot` takes a well file path. `pp.interpret(..., output="well.las")`
  writes the interpreted well, as `pp.impute(..., output=)` does, and
  `pp.impute` takes an interpretation result and returns it.
- `pp.export_context(result)` without an output path returns the context as
  a dict; `layout="package"` (CLI `--layout package`) writes the context
  package of a folder of wells.
- `calculate_pore_volume_weighted_sw`
  (`petrophys.evaluation.summaries`) gives the pore-volume-weighted Sw over
  net pay, sum(Sw PHIE h) / sum(PHIE h), with PHIE_CC preferred. The
  formation and zone tables and `summarize_pay` (`avg_sw_pay`) report this
  Sw, so average porosity x (1 - average Sw) x net pay is the hydrocarbon
  pore thickness.
- `well.curves["GR"]` returns the curve, as `well["GR"]` does; `well.curves`
  is the list of curve names.
- `petrophys.datasets.integration_fixture` downloads a Daydream 1 DLIS or
  validation LAS on first use, from `DATA_URL` (`PETROPHYS_DATA_URL`) into
  `data_dir()` (`PETROPHYS_DATA_DIR`), and checks its SHA-256, so the DLIS and
  FMI examples run after `pip install` without Git LFS. A complete
  source-checkout copy is used directly; `download=False` never fetches.
- `petrophys.reporting` exports `export_context_profile`,
  `interpret_with_context_profile`, `interpretation_group_records`,
  `context_group_record` and `context_group_statistics`;
  `petrophys.preprocessing.quality` the bad-hole flag and masking helpers;
  `petrophys.preprocessing.standardization` the user curve-name helpers. They
  are in the API reference.
- Settings and functions of the interpretation:
  - settings `rmf_chlorides_mg_l`, `carbonate_calibration`,
    `sw_routing_vcl_cutoff`, `nd_clay`, the neutron-density clay point, and
    `lithology_curve`, the lithology code curve;
  - clay volume: `nd_shale_point`, `nd_clay_separation` and `ND_CLAY_POINT`
    in `petrophys.interpretation.clay_volume.nd_method`, `sp_polarity` in
    `petrophys.interpretation.clay_volume.sp_method`, and
    `VclModule(sp_fallback_only=..., nd_clay=..., nd_triangle_only=...)`;
    `nd_clean_line` accepts matrix aliases (`ss`, `dolo`, `ls`, ...);
  - porosity and minerals: `MineralProperties.log_density`,
    `MatrixProperties.get_log_density()`, the minerals sylvite, carnallite,
    polyhalite and kainite (kainite has no published slowness),
    `total_porosity(gas_method=...)`, `steps.porosity(hc_type=...)`,
    `MultiMineralSolverV1(row_uncertainty=...)` with `ROW_UNCERTAINTY`,
    `LithologyModule(solver_options=...)` and the module
    `petrophys.interpretation.lithology.carbonate_minerals`;
  - core correction and permeability:
    `PermeabilityModule(core_porosity_curve=..., min_porosity_r_squared=...)`
    with `CORE_POROSITY_MIN_POINTS`, `PermeabilityModule(lithology_curve=...)`
    and `wyllie_rose(phi, swi, preset=..., p=..., q=..., r=...)` with
    `WYLLIE_ROSE_PRESETS`;
  - saturation: `SaturationModule(rwb_fallback=..., effective_waxman_smits=...)`,
    `SaturationResult.fallback_counts` and
    `SaturationResult.waxman_smits_clipped_to_zero`, `MAX_CEC_MEQ_PER_100G`
    in `petrophys.interpretation.saturation.waxman_smits`, and `rwb` in the
    `WorkflowConfig` of `StandardWorkflow`;
  - water and temperature: `brine_density`, `salinity_mg_per_l_to_ppm`,
    `chlorides_to_nacl_ppm` and `NACL_PER_CHLORIDE` in
    `petrophys.interpretation.electrical.rw.salinity_rw`,
    `temperature_curve_f` in
    `petrophys.interpretation.temperature.temp_profile`, and `strict=True`
    on `petrophys.model.units.temperature_unit`;
  - invasion: `InvasionModule(class_bands=...)`, its `shallow_curves` and
    `medium_curves`, and `SHALLOW_RESISTIVITY_CURVES` and
    `MEDIUM_RESISTIVITY_CURVES` in
    `petrophys.interpretation.electrical.invasion.invasion_module`;
  - NMR and saturation height: `t2_fraction_below_cutoff`, `partial_bins` on
    `bvi_from_bins` and `ffi_from_bins`, and `tvd_curve` on
    `saturation_height_from_logs`;
  - net pay and screening: `steps.net_pay(exclude_non_reservoir=...)`,
    `generate_bad_hole_flags(drho_alone=...)`, which also flags washout from
    |DRHO| alone where the caliper test cannot run (no caliper curve or no
    bit size), and `plausible_ranges` on `standardize_curves` and
    `CurveStandardizer`;
  - classes: `CLASS_TABLES`, `ClassTable`, `Bands`, `ReservoirQualityTable`
    and `ReservoirGrade` in `petrophys.model` (module
    `petrophys.model.class_tables`), with `TwoCurveClassTable` and
    `TwoCurveBands` for two-curve classes, which no canonical table uses;
    `CLASS_TABLES["RESERVOIR_QUALITY"]`, a `ReservoirQualityTable`, has
    `classify(porosity, permeability, clay, family)`, `describe(families)`,
    `grades` and `labels` but no `bands`; `petrophys.evaluation.diagnostics`
    also exports `Bands` and `bands_of`, and `bands_of` on an analyzer
    instance gives the bands it classifies with;
  - classification arguments: `class_tables` on `PorosityAnalysis`,
    `PermeabilityAnalysis`, `SaturationAnalysis`, `HydrocarbonAnalysis`,
    `LithologyAnalysis` and `run_comprehensive_analysis`, `lithology_family`
    (one family or lithology for every sample, or one per sample) on all of
    them but `SaturationAnalysis`, and `percent` on `classify_porosity`,
    `classify_sw` and the `classify` methods of `PorosityClassification` and
    `HcSaturationClassification`;
  - lithology diagnostics: `LithologyAnalysis(ground_truth_groups=...)` with
    `CURRENT_GROUND_TRUTH_GROUPS`, and `CURRENT_PROFILE_CODE_NAMES`.
- Options of the "geolink" profile, usable on their own:
  `InvasionModule(medium_deep_proxy=..., depthwise_curve_merge=...)`,
  `SaturationModule(sxo_quicklook_fallback="per_sample"|"whole_curve",
  depthwise_curve_merge=...)`, `PermeabilityModule(porosity_curves=...)` and
  `HydrocarbonAnalysis(porosity_curves=...)`. Their defaults give the
  `"current"` calculations.
- `apply_value_range_qc`, `clean_curve_values` and `rule_for_curve` take a
  `rules` argument.
- `DynamicWorkflowConfig.swirr_flag_phi_cl` chooses the PHIE behind the
  reservoir flag that picks the SWIRR_APP samples. The default `"workflow"`
  keeps the reported PHIE; `"all_shale_phit"` recomputes PHIE for that flag
  only with clay porosity from the mean PHIT of all shale-lithology samples,
  as the eval package's main run computed PHIE. The trace records the rule
  and its phi_cl.
- Raw MDT pretest pressure picking through `read_mdt_pretest` (DLIS, LAS,
  CSV or DataFrame) and `pick_mdt_pressure`, configurable buildup QC, review
  flags, and a library plot of selected windows. `formation_pressures_from_mdt`
  passes accepted picks to station interpretation with explicit TVD metadata.
  Three original MDT pretest DLIS files from IODP Hole C0009A are bundled with
  LAS copies.
- Unlabeled station-pressure contact inference through `infer_contacts`,
  exposing competing fluid hypotheses and unresolved contacts, alongside the
  labeled FWL fit.
- Public `export_context` and `petrophys-context` CLI with
  schema version 1.0, UTF-8 byte budgets, staged package publication and
  overwrite protection. The CLI always computes its context from input data,
  with the same workflow as direct interpretations. Each JSON file is held to
  `max_bytes` (1,000,000 by default); a package's total size is not limited
  unless `max_total_bytes` is set, since a package of many wells or thin
  lithology beds holds thousands of files (the GeoLink package is about
  740 MB).
- Standardized well-parameter and core-measurement readers and templates.
  Package generation accepts these alongside optional formation, lithology,
  and lithology-code tables.
- Context-package tops accept CSV/TSV/Excel, direct NPD/SODIR headers,
  explicit bottoms, depth units and rank selection without private default
  workbooks.
- `read_lithology_codes` validates codebooks and prefers Kingdom `Attribute`
  rock labels over a conflicting `Lithology Attribute` grouping, with a
  warning.
- An MCP server (`petrophys-mcp`, `pip install "petrophys[mcp]"`) that lets
  LLM clients load wells, interpret them, inspect traces, summarize zones,
  plot and export results, with file access limited to chosen folders; a
  guide and tutorial 20 cover setup and a full session. Its tools include
  `check_data_quality`, `spectral_gamma`, `run_workflow` /
  `workflow_step_kinds` (custom workflows) and `interpret_wells` (many wells
  in one table); `interpret_well` returns its assumptions and review notes,
  and `summarize_zones` takes `rank`. Paths are reported relative to the
  allowed folders.
- MCP deployment: `petrophys-mcp serve --http` (streamable HTTP or `--sse`)
  with bearer-token auth, required beyond localhost, and `/healthz`;
  `petrophys-mcp config <client>` prints the setup for Claude Code, Claude
  Desktop, Cursor, VS Code, Codex and OpenAI; `petrophys-mcp check` verifies
  folders and a live session; a Dockerfile and docker-compose.yml;
  `examples/mcp_openai_agent.py`; and `scripts/mcp_live_eval.py` for
  running a live evaluation.
- `petrophys.workflow.config_schema()` and `workflow_steps()` describe every
  setting and step as data, and the documentation site serves `llms.txt`.
  `config_schema()` entries carry the setting's `group`.
- Short public API: `petrophys.read`, `petrophys.plot` and the
  copy-by-default `petrophys.interpret`, with full docstrings;
  `DynamicWorkflowConfig` is exported at the top level, and
  `DynamicWorkflowResult` and `run_dynamic_workflow` from
  `petrophys.workflow`.
- `petrophys.exceptions.DataQualityWarning` for implausible inputs: curves mostly outside
  their plausible range (with a unit hint such as "values look 1000x too
  large"), geology that does not overlap the logged interval, and
  unrecognized lithology labels. Warnings escalated to errors (for example
  `warnings.simplefilter("error", DataQualityWarning)`) stop `pp.interpret`.
- `DynamicWorkflowResult.to_las()` writes the interpreted well with the
  settings used (`PP_*` parameters, including `PP_RW` and its temperature)
  and step reasons, without the clay-volume alias copies (VCLAY, VCLGR,
  VCLSP, VCLND); traces record the package version.
- Several wells in one call: `pp.read` and `pp.interpret` take a folder, a
  file pattern or a list of wells or files. A multi-well tops table is read
  once; each well is interpreted with the same settings, checked once before
  any well runs. The `MultiWellResult` gives `summary()` and
  `formation_summary()` as one table with a `Well` column, `to_las(folder)`
  and `save_plots(folder)` a file per well, each well's result by name, and
  the wells that failed in `failed`. `pp.plot` on several wells draws each
  well's log plot, one figure at a time (`.save(folder)` writes a file per
  well); with `curves=` it draws one figure with a track per curve and every
  well's curve in it. A mapping of labels to wells or results
  (`pp.plot({"archie": run_a, "indonesia": run_b}, curves=["SW"])`) names
  each line, so runs of one well can be compared.
- Short calls on a result: `result.step("rw")` gives one step's record
  (method, decisions, parameters), and `result.statistics(curves, top,
  bottom)` a table of any curves' statistics over a depth window, for
  `pp.interpret` and custom `Workflow` runs alike; `curve.description` reads a
  curve's description. Plotters
  (`create_nd_crossplot`, `create_pickett_plot`, ...) and the core
  calibration functions (`core_data_from_well`, `calibrate_porosity`, ...),
  `cutoff_sensitivity` and `zone_well` take an interpretation result as well
  as a well. `result.summary()` describes PHIE_CC and PERM_CC beside PHIE and
  PERM, and the several-wells `summary()` takes its PHIE and PERM medians
  from them where the interpretation produced them, as the formation
  summaries do.
- `create_histogram_comparison(wells, "GR")` compares one curve across wells
  or results; `calculate_flow_units(core)` and `create_flow_unit_plot(core)`
  take core data, and the flow-unit result gives `to_dataframe()` and
  `summary()`; `fit_j_function_from_well(well, fwl_depth)` fits a Leverett
  J-function to a well's log Sw above a free-water level, and the model's
  `plot()` draws the fit.
- Notebook display: every figure shows at the end of its cell, once,
  without `%matplotlib inline` or `plt.show()`, and `pp.plot(...)` as a
  cell's last line shows its figure, not the viewer's name. Log-plot track
  headers space their curve rows so names and units do not overlap; volume
  fractions show as v/v (saturations as frac), an imputed curve takes its
  measured curve's unit, and a track of curves no template track takes is
  titled by them. The NMR panel's permeability axis spans the data.
  `formation_summary` gives one warning for the curves undefined at some
  pay samples, and a borehole image prints a one-line summary.
- Batch use: outside an interactive session `LogViewer.save` closes the
  figure after saving, so plotting many wells keeps memory flat; a later
  `save` or `show` redraws the same depth window. `petrophys.read` on a
  folder says how to read its files.
- A curve whose samples are mostly outside the plausible range (usually a
  wrong unit label) is left out whole, not used for its in-range rest; a
  bad-hole flag that removes net pay is named in the warnings with its
  thresholds; Rmf and Rw alone make no interpretation (`status` is
  `"no_interpretation"`); `summary()["net_thickness"]` always has
  `net_rock`, `net_res` and `net_pay` (None when not computed).
- Readers: CSV and ASCII tables treat -9999 as null and take a single
  `null_values` value; `NMRData.from_csv`/`from_dataframe` match columns
  without regard to case and take `porosity_unit` (v/v porosity is converted
  to pu, with a warning when inferred); a DLIS read names requested channels
  it does not find and warns when a merged grid would take more than 0.5 GB;
  AIT-H/M (AHT90, AHO90, AHF90, ...), HRLA/HALS (HLLD, HDRS, HLLS), CNC and
  CALX are curve aliases. LAS header synonyms stand in for a missing
  standard name: the Schlumberger mud-sample names RMS, RMFS, RMCS and their
  temperatures MST, MFST, MCST for RM, RMF, RMC and RMT, RMFT, RMCT; MRT
  and BHTEMP for BHT; TDL, DL, TDD and DD for TD; SALW for SAL.
- Interpretation results checked against plots of real wells:
  - The standalone `calculate_vcl` / `VclModule` pick the GR baselines at
    the 5th and 95th percentiles, keep SP as a fallback and leave VCL_ND out
    above the N-D triangle, as `pp.interpret` does.
  - Dual Water gives Swe = 1 where bound water fills the pore space (shale)
    and NaN where the total water saturation falls below the bound-water
    saturation (inputs that disagree); Waxman-Smits takes B from the
    Waxman and Smits (1968) relation in Rw when `b_coeff` is not given.
  - The mineral solvers keep anhydrite only where RHOB exceeds 2.85 g/cm3
    with NPHI below 0.12; other samples are solved without it, so chalk
    and limestone carry no anhydrite.
  - The core porosity correction leaves log porosity at or below its fitted
    range (shale) as it is; core-calibrated permeability (PERM_CC) comes
    from the corrected porosity, on the scale of the core porosity its
    transform was fitted on, and is not extrapolated outside the rock of
    the core pairs: their porosity range, their clay volume (P95 of VCL
    at the plugs) and the net-rock clay cutoff of the run. A sample that a
    formation's own transform leaves without permeability is not filled
    from the whole-well transform. The "geolink" profile applies neither
    clay limit.
  - A zone run's whole-well tallies (bad-hole pay) are not repeated under
    the zone's label.
  - Shale takes the grain endpoints of its solids for total porosity (clay
    -bound water is porosity, as phi_cl counts it), not the wet-shale bulk
    density; a sample labelled coal whose density reads 2.0 g/cm3 or more
    takes the fallback matrix, not coal's 1.35 g/cm3.
  - Net reservoir and net pay leave out layers thinner than the density
    log resolves (about 0.46 m) between two halite readings, and coal read
    from the logs (RHOB below 1.8 g/cm3, NPHI above 0.35 where logged, and
    the samples within 0.23 m of it) where no lithology group calls it coal.
  - Coates and the other Swirr-based permeabilities take Swirr from the
    Buckles bulk volume of water: BVWirr / PHIE at each depth, with BVWirr
    the median PHIE x SW of the hydrocarbon-bearing reservoir.
  - The temperature trace names the total depth `td` with its `td_unit`;
    Rcl is printed rounded.
- Evaluation checked against the logs it summarises:
  - Per-formation and per-zone tables average over net pay the porosity
    and permeability the interpretation used at each sample: the
    core-corrected curve where the run wrote it, the plain one at the other
    samples (`pay_average_data`); `df.attrs["curves"]` names them and
    `df.attrs["plain_curve_pay_samples"]` counts, per formation, the pay
    samples read from the plain curve. Porosity
    and clay volume are weighted by thickness, Sw by pore volume and
    permeability is a geometric mean, so `Avg PHIE x (1 - Avg Sw) x Net Pay`
    is the `HCPV` column (hydrocarbon pore thickness). Tables give
    `N/G (reservoir)` (net reservoir / gross) and `Pay/Gross` (net pay /
    gross).
  - `result.summary` and `result.statistics` read a depth window in either
    order and warn when it holds no sample of the well.
  - `pp.interpret` warns when a run ends as "failed" or without porosity or
    saturation for lack of a log, naming the missing logs, the curves that
    match no standard name and the `curve_aliases` setting that maps them.
  - `well.depth_slice` and `well.select` keep the bottom-hole temperature of
    a later interpretation at the deepest log sample of the well they were
    cut from.
  - `run_formation_analysis` attaches the tops before interpreting, so it
    matches `pp.interpret`; `monte_carlo` reports `correlated_depths`
    inputs in `sensitivity`; `run_comprehensive_analysis` keeps curves the
    well already has (BAD_HOLE, CROSSOVER_FLAG) and lists them; QuickLook
    uses its `vcl_cutoff`, `rt_cutoff` and `min_thickness` (in the well's
    depth unit) and averages per-sample Sw over pore volume; FZI is NaN
    below 0.01 effective porosity.
  - Diagnostics: two of three clay indicators make a high clay class; the
    carbonate check without PEF or RHOB is "undetermined"; hydrocarbon labels
    say water-bearing or undetermined where that holds, with no default
    "marginal"; BVW
    without effective porosity has its own class; trend classes scale by
    each curve's P10-P90 range; geometric and harmonic means are 0 with a
    zero value and NaN with a negative one. The benchmark profiles keep
    their classes.
  - The MCP well-loading reply names the files it attached.
- Specialist data: `eaton_from_well` takes a `shale=` mask (Eaton is a
  shale method), leaves samples that would read below zero blank and notes
  them, and calibrates its exponent against the shale-trend value at each
  pressure depth, refusing an exponent that is not positive;
  `CoreData.from_dataframe`/`from_csv` take `porosity_unit=` and
  `saturation_unit=` and convert percent values with a warning;
  `kozeny_carman_perm`, `sorting_coefficient` and `skewness` say why they
  give no result; `create_pressure_contact_plot` frames the fitted stations
  (`zoom="fit"`, or `"all"`); the MDT pretest plot wraps its review flags
  and shades the mud baselines it tested; clay typing labels samples it
  cannot type "unclassified".
- Reading and units: a tops table's `depth_ref` column is read by its
  values (KB, RKB, DF, GL, MSL ... as the datum; MD, TVD, TVDSS as the depth
  reference) and its datum is checked against the LAS log datum (LMF);
  percent porosity curves of other packages (DPOR, DPHI, DPHZ, SPOR, ...)
  are converted to v/v in standardization; `export_csv` writes headers as
  `NAME (unit)` and the CSV/ASCII readers and `pp.read` take a curve's unit
  (and the depth unit) from such headers; a range-screening warning names
  the source channel and frame of a merged DLIS curve.
- Reading and QC: a time-indexed DLIS frame (a pressure pretest) is not
  merged into a depth log, and a file of such frames is refused with a
  pointer to `read_mdt_pretest`; an absolute caliper named DCAL takes the
  caliper range in value-range QC; `histogram_matching` maps each sample
  by its rank to the reference quantiles.
- Plots checked track by track on real wells:
  - Fixed templates leave out tracks with no drawable curve and list them
    in `omitted_tracks`; a curve mostly off its track's scale is named in
    `viewer.warnings`; DTE is drawn in the sonic track; BVW shares the
    porosity scale in every template; the "raw" template and the triple
    combo take the well's own SP range; `geolink_full` is another name for
    the `comprehensive` layout.
  - The lithology legend has an entry for every colour and pattern drawn;
    a formation top at the bottom of the window is not labelled below the
    plot; `curve_styles` `display_name` applies to resistivity and alias
    curves.
  - `create_multiwell_overlay`, `plot_core_on_logs` and `create_log_plot`
    find curves by standard name, convert units and depths, and draw
    permeability on a log axis; the tornado chart shows both bars when they
    fall on one side of the base; the Buckles colour bar spans the 1st to
    99th percentile of permeability; zone flags and the net-pay summary use
    the log viewer's flag colours.
- Plots: after `save` in a script the figure is released from pyplot but
  kept in `.fig`, so edits survive later saves; the multi-well comparison
  converts curves to standard units, reverses neutron and sonic and warns
  when units still differ.
- Core and electrical analysis: `apply_poro_perm_transform` writes
  the core-calibrated `PERM_CC` (a core-plug name such as `PERM_CORE` would be
  read back as plugs), and `PERM_CORECAL` is an alias of `PERM_CC`; `plot_core_on_logs` overlays plugs on any PHI*, PERM*, SW* or SO*
  curve and `plot_poro_perm` draws the transform over the plugs' porosity
  range; `find_core_depth_shift` warns on a weak tie (r < 0.5);
  `pickett_analysis` returns Rw and warns when m lies outside 1.3-3; it
  fits by reduced major axis (errors in both porosity and resistivity),
  with `method="ols"` available and m fitted through log(a x Rw) when `rw=`
  is given, and `PickettPlot.add_trend_line` draws the same line.
- Context export of a user's well clips permeability to the 0.0001 mD QC
  floor, so tight rock stays in the statistics; its
  methodology text names the porosity order it uses. `petrophys-context`
  takes `--setting NAME=VALUE` (repeatable) for interpretation settings.
  A custom `WorkflowRunResult` can be pickled.
- Every skipped step records a specific reason, and `print_trace()` shows it.
- `formations_costa.csv` labels the logged interval of each Costa well as
  its single formation, KHARAIB-2, so formation summaries and context
  packages name the reservoir.
- `DynamicWorkflowResult.formation_summary()` tabulates gross, net and
  average properties per formation. Tops that mix levels give formations
  unless `rank=` says otherwise, tops at or below the logged bottom are left
  out, and NPD informal units ("Intra Balder Fm. Sst.") rank as members.
- NPD/GeoLink-style tops without a rank column get their stratigraphic rank
  from the name suffix (GP./Fm./Mb.), so mixed group/formation picks can be
  analyzed one level at a time with `run_formation_analysis(..., rank=...)`;
  `read_formations(..., rank=...)` keeps one stratigraphic level.
- Formations attached with `petrophys.read(..., formations=...)` calibrate
  core permeability per formation when they form a single level.
- Shared formation import for the model, viewer, and formation analysis,
  with explicit column, worksheet, well, depth-unit, reference and datum
  handling. Invalid geology rows raise, summary totals count the covered
  depth union, and unknown reader or workflow options raise. Top-only
  analysis takes `terminal_depth`.
- External lithology precedence, numeric category metadata, and gap/unknown
  reports; core saturation fraction normalization with provenance
  (`petrophys.model.units.core_saturation_to_fraction`).
- Custom workflows: `Workflow` and `workflow_step` (`petrophys.workflow`) build reusable
  interpretation recipes from built-in steps (`petrophys.workflow.steps`,
  including `pp.interpret` as one step) and your own functions. Inputs are
  checked before running; runs keep a full trace and export to LAS/JSON;
  `run_many` runs a field; workflows save and load as JSON. Guide page and
  tutorial 21.
- Analysis tools: `find_core_depth_shift` / `shift_core_depths` (core-to-log
  depth matching, with per-core-run scans); `find_held_readings` /
  `held_reading_mask` (held or padded tool readings); `summarize_pay` /
  `PaySummary` (net pay, NTG, pay averages and hydrocarbon pore thickness
  for a window); the `petrophys.evaluation.sensitivity` module with
  `cutoff_sensitivity`, `parameter_sensitivity` (tornado data) and
  `compare_saturation_models`; `sp_rw_in_window` (SP-derived Rw in a chosen
  window at a reference temperature); `classify_th_u_environment`,
  `thorium_clay_index` and `uranium_stripped_gamma_ray` for spectral gamma;
  `label_depths` and `flag_intervals`.
- `pp.impute(well, curve, train_wells=None, output=None)` reconstructs a
  curve's missing samples. A gradient-boosted tree model (scikit-learn
  `HistGradientBoostingRegressor`) is trained on the fly on the other logs
  wherever the curve is measured, in the well and in `train_wells`, so a
  curve the well lacks altogether can be rebuilt from offset wells. The
  result is added as `<CURVE>_IMPUTED`, with the measured values kept, and
  written to a LAS file with `output=`. A training well without one of the
  input logs still teaches the others (its missing log is a gap the trees
  route), and held readings of the curve (runs of 10 or more identical
  samples, such as a zero-filled curve) are not learned from. A warning
  counts imputed samples whose inputs lie outside the training range. Needs
  the `ml` extra. `pp.plot` draws `<CURVE>_IMPUTED` in the track of the
  curve it rebuilds, on its scale, dashed where it fills that curve's gaps.
  Curves of one unit in a generic track share one scale. `<CURVE>_PREDICTED`
  checks the model: a prediction at every depth, where each measured sample
  is predicted by a model that did not learn from its fifth of the well (five
  contiguous depth blocks), so its difference from the measured curve is the
  error to expect in a gap. The curve's description gives that error (mean
  absolute error, bias, R2) in the curve's unit, or in log10 of the curve
  for resistivity and permeability, and `pp.plot(well, curves=["RHOB",
  "RHOB_PREDICTED"])` draws it over the measured curve. A training well
  that is the well itself is left out, with a warning, so the check does
  not learn from the samples it predicts.
- Plotting: `create_heatmap` centres each cell on its coordinate, draws
  unevenly and log-spaced bins such as NMR T2, and accepts cell edges; the
  core plotting helpers (`plot_calibration`, `plot_poro_perm`,
  `plot_core_on_logs`) and `GEOLINK_LITHOLOGY_CODE_LABELS` are exported from
  `petrophys.visualization`. The log viewer reads the lithology codebook from
  `lithology_mapping` or `PETROPHYS_LITHOLOGY_MAPPING`.
- Public modules are organized under `model`, `io`, `preprocessing`,
  `interpretation`, `evaluation`, `workflow`, `reporting`, and
  `visualization`.
- Documentation: an API reference with a landing page per area (front door,
  data model and I/O, preprocessing, interpretation, evaluation, workflows,
  visualization, reporting and LLM agents, and the fixed-configuration
  `StandardWorkflow`), a page per section with
  grouped summary tables, and one page per function and class, generated
  from each package's `__all__`. The user guide pages are executed at each
  documentation build: quickstart, own-data, input templates, data model,
  glossary, architecture, workflows, review, context, MCP, visualization,
  specialist data and capabilities, with "What Petrophys can do" as a
  gallery of every capability. An interpretation-settings page generated
  from `config_schema()`, a "Terms, curves and steps" reference and a review
  checklist for interpretations.
- Getting started for people who have not programmed: "New to Python?"
  explains notebooks, running cells and reading your own file in Google
  Colab, and "On your own computer" installs Python, Petrophys and
  JupyterLab step by step for data that must stay on the machine.
- Tutorials: a complete-tour notebook covering the whole library on four
  bundled GeoLink wells, with COSTA and Daydream 1 data for the specialist
  parts, foundations and advanced series, case studies, and a
  spectral-gamma geochemistry part and a second-well core calibration in
  tutorial 12. Each tutorial and the quickstart opens in Google Colab from an
  "Open in Colab" button, with a first cell that installs Petrophys there;
  `scripts/colab_notebooks.py` adds the buttons and writes the quickstart
  notebook from the quickstart page. `scripts/check_notebook_outputs.py`
  rejects tutorial notebooks with unexecuted cells, errors or local paths in
  their saved outputs.
- Third-party data provenance and model validation records accompany the
  package.

### Known limitations

- A whole-well run uses one Rw, corrected for temperature at a constant
  salinity, although shallow formation waters are usually fresher, and the
  clay indicators disagree in undercompacted mudstones. Default runs can
  therefore flag marginal pay (SW about 0.5) in shallow Tertiary sections
  that no lithology code log covers: on 25/5-1, 199 m of the 251.0 m of net
  pay lies above 2984 m, nearly all of it uncoded. Read the results per
  formation (attach tops and use `result.formation_summary()`), or give the
  shallow interval its own Rw (a zoned header parameter such as `RW_Z1`
  between `TOP_Z1` and `BOT_Z1`, which
  `petrophys.workflow.run_dynamic_workflow_with_zones` applies).
  Petrophys does not model an Rw that varies continuously with depth.
- Clay porosity and the neutron-density clay point are each one value per
  well, picked over the whole log, so they do not follow shale compaction
  with depth; set `phi_cl` or `nd_clay` from the interval you evaluate.
- With oil-based mud and no Rw input, Rw comes from the clean-wet Rwa
  estimate, with a warning, which assumes that some clean rock in the well
  is water-bearing; pass `rw`.
- The generic carbonate solve takes calcite and dolomite from bulk density,
  so where porosity is density-derived the split largely follows
  `matrix_type` and `rho_fl`; use `carbonate_calibration="costa"` for the
  COSTA wells.
- `steps.interpret` does not apply the LAS header parameters that
  `petrophys.interpret` turns into settings (such as CLFL, A, M, N or the
  cutoffs); pass them as settings.
- The diagnostics analyzers and `run_comprehensive_analysis` grade reservoir
  quality with clastic limits unless `lithology_family` is given, also on a
  carbonate well; the workflow and context exports pass each sample's
  family.
- A lithology interval label that the workflow does not map, such as
  "sandy limestone" or "calcareous shale", takes the run's lithology family
  in net pay but the family of its last rock word in a context's
  reservoir-quality grades, so there FAIR_RESERVOIR can disagree with
  NET_RES. Lithology codes are not affected.
