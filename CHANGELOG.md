# Changelog

All notable changes to Petrophys are documented here.

## 3.0.0 (unreleased)

Release 3.0.0 corrects the science of the default interpretation and keeps
the calculations behind the GeoLink eval context benchmark package as a
calculation profile that regenerates the package from its raw logs. Default
results change: "Corrected calculations" lists each correction and its
effect on the bundled wells.

### Calculation profiles

- The calculation profile that keeps the GeoLink eval package's calculations
  is named `"geolink"` (it was `"benchmark"`), like the context profile that
  reproduces that package: `calculation_profile="geolink"`,
  `petrophys-mcp serve --calculation-profile geolink`,
  `PETROPHYS_CALCULATION_PROFILE=geolink`, and `class_tables="geolink"` for
  the diagnostics analyzers. `"benchmark"` is refused with a message naming
  the new value.
- The published GeoLink, COSTA and Danish context packages are computed by
  context profiles, `geolink_v1`, `costa_v1` and `danish_v1` (aliases
  `geolink`, `costa`, `danish`), each running its package's own pipeline from
  the delivery's raw LAS files with the calculations the package used:
  `export_context(..., context_profile=...)`, `petrophys-context --profile ...`,
  and `petrophys.reporting.context_profiles.export_context_profile`. Without a
  profile, the current workflow runs. A profile fixes its package's settings
  and rejects `workflow_config` and the other interpretation options. The
  GeoLink profile reads the raw `eval` LAS, the NPD tops workbook and the
  GeoLink lithology codebook. The COSTA
  profile needs no files besides petrophys: the COSTA LAS, the per-well
  interpretation parameters and the reference calcite/dolomite volumes are
  bundled (`datasets/costa`, `_internal/data`). The Danish profile reads the raw LAS, the Danish
  tops workbook and the GeoLink lithology codebook. The interpreted LAS of
  both packages are reproduced byte for byte, and every context number and
  note except the ELECTROFACIES statistics.
- `scripts/generate_benchmark_context.py` accepts one LAS file or a folder
  from any eval collection and recognizes GeoLink, COSTA or Danish wells by
  name (`--dataset` overrides); for COSTA and Danish it also keeps the
  interpreted LAS (and the prepared Danish LAS).
- The category context exporter takes `group_layout`: per well, the
  formation/lithology groups to export instead of splitting by tops and
  lithology codes (the Danish profile's one group per formation).
- Under `"geolink"` the interpreted LAS keep the source's depth unit label
  (e.g. `FT`) and equivalent unit labels (`gAPI`, `m3/m3`), write the
  temperature parameters without units, and pad header columns by UTF-8 bytes,
  as the eval package's files were written; `export_las` takes
  `depth_unit_label` and `pad_header_by_bytes` for this. The GeoLink package's
  interpreted LAS of 15/9-13, 15/9-17, 30/6-5, 34/10-21, 34/10-23 and 35/9-2
  are now reproduced byte for byte too.

- `DynamicWorkflowConfig.calculation_profile` chooses the calculations.
  `"current"`, the default, is the maintained interpretation, with the
  corrections listed under "Corrected calculations". `"geolink"` keeps the
  calculations behind the GeoLink eval context benchmark package fixed, so
  the package can be regenerated from its raw LAS files. The setting's
  documentation lists every difference between the two.
- The package was made in two runs with different calculations: its main
  run interpreted 30 of the 32 wells, and a re-run interpreted 15/9-17 and
  34/10-23 and wrote the context of every well. Under `"geolink"`,
  interpreting a well (`petrophys.interpret`, `run_dynamic_workflow`,
  `DynamicWorkflow`, the zoned batch runner, the MCP server) uses the main
  run's calculations, except for 15/9-17 and 34/10-23, which get the
  re-run's. Either well is recognised by its UWI, its WELL header or, in the
  zoned batch runner, its LAS file name, compared without case or
  surrounding spaces and with `/` read as `_`; `15/9-17 A` is another well.
  Writing context under `"geolink"` uses the re-run's calculations for
  every well. The workflow trace names the calculations a well got in
  `config.benchmark_calculations`: `"main_run"` or `"rerun"`.
- The main run's calculations differ further from the re-run's: they take
  clay porosity from the mean PHIT of every shale-lithology sample; read
  PHIE and PERM rather than the core-corrected PHIE_CC and PERM_CC, with no
  PHIT fallback; use one Archie coefficient set from the lithology domain
  rather than per-depth values; use the SXO quick-look relation only as a
  whole-curve fallback; take the invasion ratio from the first flushed-zone
  or shallow resistivity present, without the medium-resistivity proxy;
  compute MHI wherever SW and SXO are finite; and write a duplicate
  BADHOLEFLAG curve.
- `petrophys.interpret`, `export_context` and the MCP server use `"current"`
  unless told otherwise; `petrophys-context` does too unless a context
  profile is selected. To reproduce the package, use `context_profile="geolink"`;
  to apply its calculations to other wells, pass
  `workflow_config=DynamicWorkflowConfig(calculation_profile="geolink", reuse_input_resflag=True)`
  to `export_context`: the package picked SWIRR_APP from the reservoir flag
  already on its interpreted LAS files.
- `petrophys-mcp serve --calculation-profile {current,geolink}`, or the
  `PETROPHYS_CALCULATION_PROFILE` environment variable, sets the profile of
  the MCP server's interpretations whose settings name none, so that an
  evaluation harness can pin `"geolink"`. A profile in a tool's settings
  still wins, `petrophys-mcp check` reports the active one, and
  `create_server` takes it as `calculation_profile`. Only the MCP server
  reads the variable; the library ignores it.
- `scripts/generate_benchmark_context.py` regenerates the package. With
  `--from-raw` it starts from the raw LAS files and writes both outputs: the
  interpreted ("processed") LAS files under `<output-dir>/processed_las`,
  then the context built from those. `--calculation-profile` (`geolink`,
  the default, or `current`) applies to both steps, and `--well` limits the
  run to the wells named. With the package's NPD tops workbook and lithology
  codebook next to the raw LAS folder (or given with `--tops-file` and
  `--lithology-mapping-file`), one command reproduces the whole package:

  ```bash
  python scripts/generate_benchmark_context.py --from-raw --input-dir PATH/eval --output-dir NEW_DIR
  ```

  It rebuilds the interpreted LAS files of all 32 wells, 31 of them byte for
  byte (16/10-4's differ at three SW samples), and the 25,566 context files.
  The 30 wells of the main run reproduce every number of the
  package apart from the ELECTROFACIES statistics, which come from k-means
  clustering that was unseeded when the package was made; 15/9-17 and
  34/10-23 reproduce every number except the SO statistics their archived
  files carry, and some of their note wording differs.
- `config_schema()` lists `profile_defaults` for the settings whose default
  depends on the profile: `standardize_input`, `vcl_cutoff` and `sw_cutoff`.
- Context exports, and the hydrocarbon analysis and invasion statistics of
  the workflow, classify with the class tables of their profile: `"current"`
  with the canonical tables of `petrophys.model.CLASS_TABLES` (see
  "Classes and reservoir quality"), `"geolink"` with frozen copies of the
  eval package's tables, rules and threshold texts. The diagnostics
  analyzers and `run_comprehensive_analysis` take the choice as
  `class_tables`, `"current"` by default.

### Corrected calculations

The default profile, `"current"`, makes the corrections below to the eval
package's calculations, and `"geolink"` keeps the package's calculation in
each case; the `calculation_profile` documentation also lists where the
package's main run differs further. A bullet that names a library function
describes that function for every caller; under `"geolink"` the workflow
keeps the package's behaviour wherever the package's calculations used the
function.

Net pay from `petrophys.interpret` with default settings on the bundled
wells (30/6-5 with `rw=0.07, rw_meas_temp_f=230`), with these corrections
and before them, that is with the package's re-run calculations
(`calculation_profile="geolink"` gives these wells the main run's
calculations instead: 1400.9 m of net pay on 30/6-5):

| Well | Net pay | Before | Notes |
| --- | --- | --- | --- |
| 30/6-5 | 49.7 m | 1279.9 m | 25.5 m inside the NPD oil column (2841-2907 m); 21.3 m above 1500 m, against 1080.7 m before |
| 16/7-1 | 2.1 m | 338.6 m | 284 m of the earlier figure lay in the salt (halite and anhydrite codes); none does now |
| 15/9-14, a dry well | 48.2 m | 828.7 m | |
| 25/5-1 | 310.4 m | 365.8 m | 259 m lies above 2984 m; see "Known limitations" |
| HW-3, COSTA carbonate, with `lithology_domain="carbonate"` | 195 ft | 195 ft | unchanged |

#### Inputs

- Input curves are standardized by default: vendor mnemonics get canonical
  aliases, units such as kg/m3 or percent porosity are converted, and
  implausible values are removed with a `DataQualityWarning`. The screening
  also removes compressional slowness below 38.5 us/ft, faster than any
  rock matrix, and SP beyond +/-500 mV: on 16/7-1, about half of the DTC
  samples and a -6183 mV SP spike.
- TEMP and TMPC curves are read in the unit their label gives (the "geolink" profile
  reads them as degF), and `standardize_curves` converts a degC or K
  temperature curve to F and screens it on its own scale.
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
- Under `"current"` the workflow finds the lithology code curve by one rule:
  the curve named by the new setting `lithology_curve`, else
  `LITHOLOGY_GEOLINK`, else, with `lithology_source` `"auto"` or `"vendor"`,
  the first curve under a lithology-code name (`LITHOLOGY`, `LITHOFACIES`,
  `FACIES`, `LITH_CODE`, `ROCK_TYPE`, ...) whose values pass the code-log
  checks: 20 or more finite samples, whole numbers from 0 to 99 with at most
  50 distinct values, half or more of them GeoLink codes. That one curve
  gives the vendor lithology, an `"auto"` matrix and the per-depth lithology
  groups behind the net-pay cutoffs and exclusions. The run records it on
  the well, and context exports, the diagnostics, the zoned batch runner,
  `steps.net_pay`, `PermeabilityModule`, `core_data_from_well`,
  `IntervalAnalyzer` and the log viewer read the same curve. A curve other
  than `LITHOLOGY_GEOLINK` is named in a warning saying that its codes are
  decoded with the GeoLink/Kingdom table; the trace gives the lithology
  source as `vendor_lithology_log[MNEMONIC]`, and its `input_context` step
  reports the curve as `vendor_curve`. Names that only contain LITH or
  FACIES, such as `LITHO_PATT` in the Daydream 1 logs, are no longer read as
  lithology codes, and curves the workflow writes (`LITHO`, `LITHO_CLASS`,
  `MMQC`, `GT_*`) never are. Unless `lithology_curve` is set, the "geolink" profile
  takes the vendor lithology and its matrix from any curve named like a
  lithology code log, and the per-depth groups from `LITHOLOGY_GEOLINK`
  alone.

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
  size. The "geolink" profile assumes an 8.5 in bit where a well has no BS curve or
  header bit size, uses a header bit size whatever its value and never to
  fill gaps in a BS curve, and tests |DRHO| only in a well with a caliper.
- Among the 32 wells of the eval package, the assumed bit size flagged
  samples only in 15/9-13, 15/9-4 and 15/9-5, which have no BS curve: their
  BAD_HOLE falls from 82-87% of the samples to 0.8-8.6%. The bundled 30/6-5,
  16/7-1, 25/5-1 and 15/9-14 have complete BS curves and are unchanged.

#### Clay volume

- VCL is the minimum of VCL_GR and VCL_ND. VCL_SP from baselines picked over
  the whole log (P5 and P95) is reported but left out of the minimum, and
  fills only depths with no GR or neutron-density value: such baselines
  mostly measure the drift of the SP shale baseline, and on 30/6-5 they read
  sands as shaly and shales as clean. Setting both `sp_min` and `sp_max`
  puts VCL_SP back into the minimum.
- VCL_ND is measured from the clean line of the run's matrix, sandstone by
  default, instead of always the limestone line. An unresolved
  `matrix_type="auto"` uses its domain's matrix or sandstone, and the run
  warns when the matrix later resolves to one with another clean line.
- The VCL_ND clay point is picked in the well's good-hole shales: the median
  RHOB and NPHI of at least 20 shale-lithology samples, else of the samples
  with VCL_GR above 0.9, else (2.62 g/cm3, 0.38). A point less than 0.10 v/v
  of neutron above the clean line leaves VCL_ND out of VCL, with a warning.
  The setting `nd_clay` sets the point instead; zoned runs use the
  whole-well point.
- VCL_ND is undefined where NPHI lies above both the clay point and the
  clean line's 40 p.u. point, outside the neutron-density triangle; VCL
  comes from the other indicators there.
- Shale-coded samples of 30/6-5, 25/5-1 and 15/9-14 now have a median VCL of
  1.0, where the "geolink" calculations give 0.24-0.62, low enough on 30/6-5
  for shale to pass the 0.3 clay cutoff.
- `sp_min` is the shale SP baseline and `sp_max` the clean-sand line; the
  setting descriptions had them the other way round. User baselines are
  checked against the SP polarity read from the SP-GR rank correlation
  (`sp_polarity`): an order that contradicts it is warned about, and equal
  values are rejected.

#### Porosity and core correction

- Evaporites take their apparent density-log reading as the density-porosity
  endpoint instead of their grain density: halite 2.032, anhydrite 2.977 and
  gypsum 2.351 g/cm3 (Schlumberger, 1972), so the halite of 16/7-1 reads
  zero porosity where it read a PHIE of about 0.09. The potash salts have
  endpoints of their own (the new minerals sylvite, carnallite, polyhalite
  and kainite), which GeoLink codes 27 and 32-34 and the labels "Sylvite"
  and "Potash" map to.
- Halite is also recognised from its logs where no lithology code calls a
  sample an evaporite: RHOB 1.95-2.15 g/cm3, GR below 15 API, deep
  resistivity above 500 ohm.m (without one, a caliper 0.5 in under the bit
  size) and, where logged, NPHI up to 0.05 and PEF 4.0-5.5. Such samples
  take the halite endpoints and are left out of net reservoir and net pay.
- Clay porosity (`phi_cl` in PHIE = PHIT - VCL x phi_cl) comes from the
  shale samples recomputed on the reservoir matrix endpoints, the basis of
  the PHIT it is subtracted from, instead of the wet-shale endpoints, which
  left PHIE in shaly sand 4-5 p.u. too high. Dual Water uses the same value.
- A LAS header PNSH, the neutron log's reading in shale, gives the clay
  porosity together with RBSH (shale density) as a neutron-density shale
  point. PNSH alone is set aside, as it is not a clay porosity; the
  "geolink" profile takes it as `phi_cl`.
- GeoLink code 10 ("Porous Limestone") takes limestone endpoints and code 26
  ("Gap", no lithology recorded) no lithology group, where the "geolink" profile
  groups them as sandstone and evaporite; context exports name them so.
- In neutron-density crossover (PHID > PHIN + 0.05), PHIT is the
  root-mean-square of PHID and PHIN (Gaymard and Poupon, 1968) only with
  `hc_type="gas"`; in oil wells it is the density porosity.
- Coal takes the chartbook slowness of 120 us/ft (bituminous coal reads
  110-140 us/ft) and the gypsum group the table's 52.5 us/ft, where the
  "geolink" profile uses 250 and 83 us/ft.
- The same-well core porosity correction is fitted to PHIT, since the helium
  porosity of cleaned, dried plugs is total porosity (API RP 40, 1998), and
  PHIE_CC = PHIT_CC - (PHIT - PHIE), kept within 0 and PHIT_CC. A fit is
  used only with at least 8 core-log pairs, a positive slope and an r^2 of
  at least 0.5, for the whole well or a formation; a rejected fit is listed
  with its reason in the trace (`core_rejected_models`). The fit of 25/5-1
  (r^2 0.08) fails the gate, so its porosity is not core-corrected.
- Library functions: `MineralProperties.log_density` and
  `MatrixProperties.get_log_density()` give the evaporite endpoints, which
  `PorosityModule` and `steps.porosity` use. `total_porosity` takes
  `gas_method` (`"rms"`, the default, or `"density"`) and counts a negative
  neutron as 0 in the root-mean-square; `PorosityModule` follows its
  default, and `steps.porosity` takes `hc_type` (`"oil"`, the default, keeps
  density porosity in crossover). `sonic_porosity` applies the compaction
  factor before the 0-0.5 limits (Hilchie, 1978). Oil and heavy oil have a
  hydrogen index of 1.0 (0.8 and 0.7 before), so the opt-in hydrocarbon
  correction leaves NPHI unchanged for oil. The mineral table's clay U is
  9.81 b/cm3 (3.7 x 2.65). `calculate_elastic_properties` returns floats for
  scalar inputs, and `ArchieParameterDefaults` reads "porous_limestone" as
  limestone and "gap" as generic clean rock.

#### Water and mud

- The header CLFL is read as mud-filtrate chlorides in mg/l, the mud-report
  unit (ppm, kppm and % labels are converted), into the new setting
  `rmf_chlorides_mg_l`. Unless an Rmf or `rmf_salinity_ppm` is given, Rmf
  comes from the NaCl equivalent, Cl x 58.44/35.45, turned into ppm with the
  brine density of McCain (1991); the trace shows each step. The "geolink" profile
  reads CLFL as NaCl ppm. On 15/9-14 and 16/7-1, whose headers give CLFL and
  no Rmf, Rmf is about a third lower, and an Rw derived from the SP falls in
  proportion; 30/6-5 and 25/5-1 take Rmf from their headers.
- The mud-sample temperatures MFST and MST stand in for RMFT and RMT as the
  measurement temperatures of Rmf and Rm.
- With `mud_type="OBM"` no SXO or MHI is computed, and Rw is not estimated
  from the SP or from Rxo/Rt: they need a conductive, water-based filtrate,
  and the 200 ohm.m Rmf written for oil-based mud is a placeholder. Rw then
  comes from `rw`, the header or the clean-wet Rwa estimate.
- Temperature helpers accept any recognised unit label, such as "degF",
  "deg C" or "K", and raise `ValueError` on an unknown one (some of them
  misread labels other than a bare "F" or "C"): `arps_equation`,
  `correct_resistivity_temperature`, `calculate_temperature_gradient`,
  `TemperatureProfile` and its `from_bht`, `from_gradient_c_per_km` and
  `from_two_points`, `rw_from_salinity`, `salinity_from_rw`, `rw_from_ssp`
  and `estimate_nacl_resistivity`. `estimate_gradient_from_bht` and
  `rw_at_reference_temperature` convert their default reference
  temperatures (68 and 75 degF) to the caller's unit, `RwModule` and
  `RmfModule` read TEMP and TMPC by their unit, and `convert_units` and
  `convert_curve_units` convert between temperature scales.

#### Invasion

- The workflow's INVASION_RATIO reads, sample by sample, RXO (the
  flushed-zone microresistivity reading), else the shallow RSHA and then
  RESS, else the RMED/RESM medium/deep proxy; the trace records the order
  (`rxo_curve_order`) and a decision naming the samples each curve gave. The
  "geolink" profile reads RSHA first, then RXO and RESS. On 15/9-14, the only
  bundled GeoLink well with RXO, the ratio changes on 20% of the samples and
  its Rxo/Rt class on 9.3%. `InvasionModule` also reads RXO first by default
  (see "Breaking changes").

#### Saturation

- The Indonesia equation is the published one of Poupon and Leveaux (1971),
  1/sqrt(Rt) = [Vcl^(1 - Vcl/2)/sqrt(Rcl) + phi^(m/2)/sqrt(a Rw)] Sw^(n/2),
  in `petrophys.interpret` and `steps.saturation`. The "geolink" profile's variant
  applies Sw^(n/2) to the sand term only, which drives shaly samples to
  SW = 0.
- Dual Water runs only with a supplied bound-water resistivity: `rwb`, a
  header RWB or `steps.saturation(rwb=...)`. Rwb is not estimated (the
  "geolink" profile takes 0.3 Rw). With `sw_model="auto"`, shaly carbonate samples
  without it take the clastic routing (Waxman-Smits with CEC and B,
  Simandoux at a salinity of 20,000 ppm or more, otherwise Indonesia), and
  the trace counts them; `sw_model="dual_water"` without Rwb stops the
  saturation step with an error that asks for it. The equation is the
  wet-zone form of Clavier, Coates and Dumanoir (1984), with the bound-water
  saturation capped at 0.999.
- Waxman-Smits (Waxman and Smits, 1968) takes `cec` in meq/100 g, as
  documented, and converts it to the meq/g of the equation; it was passed
  on unconverted, which gave a hundred times the clay conduction. A `cec`
  outside 0-150 meq/100 g (pure smectite is 80-150) is rejected, and one
  below 1 is warned about as a likely meq/g value. `sw_model="waxman_smits"`
  without `cec` and `b_coeff` gives a clear error.
- The Waxman-Smits saturation, solved on total porosity, is reported on the
  effective basis, SW = 1 - (PHIT/PHIE)(1 - Swt), so BVW, HCPT and net pay
  pair it with PHIE like every other model. Where SW is paired with PHIE_CC,
  Waxman-Smits and Dual Water take PHIE_CC + (PHIT - PHIE) as total
  porosity. Samples whose Swt lies below the bound-water fraction
  (PHIT - PHIE)/PHIT get SW = 0 and are counted in the trace.
- The new setting `sw_routing_vcl_cutoff` is the clay volume above which
  `sw_model="auto"` uses a shaly-sand equation. Unset, it is `vcl_cutoff`,
  or 0.3 when that is unset too, so net rock with 0.3-0.4 clay (below the
  clastic standard of 0.40) gets the shaly-sand equations. The SWIRR_APP
  pick uses the same cutoff.
- Library functions: `SaturationModule` no longer estimates Rwb as 0.3 Rw
  (`rwb_fallback=None` by default; pass `rwb`, or `rwb_fallback=0.3` for the
  old rule), and `effective_waxman_smits=True` reports the Waxman-Smits Sw on
  the effective basis. `SaturationResult` gains `fallback_counts` (for
  example `{"dual_water->indonesia": 12}`) and
  `waxman_smits_clipped_to_zero`. `calculate_sw` warns with the number of
  fallback samples, and `StandardWorkflow` takes `WorkflowConfig.rwb`.

#### Mineral solvers

- `MultiMineralSolverV1`, the clastic workflow's solver, divides each
  equation row by its nominal one-sigma tool uncertainty before the
  non-negative least squares, the weighted solve of Mayer and Sibbit (1980)
  and Quirein et al. (1986): `row_uncertainty`, default `ROW_UNCERTAINTY`
  (closure 0.01 v/v, RHOB 0.02 g/cm3, NPHI 0.02 v/v, U 0.5 b/cm3, DT
  2 us/ft). Unweighted, the sonic and U rows dominate the fit and the 0.05
  closure gate rejects most samples whose logs agree within tool precision;
  `row_uncertainty=None` keeps that solve, which the "geolink" profile uses. The
  mineral volumes change; porosity, saturation and net pay of the bundled
  wells do not.
- The carbonate domain takes calcite and dolomite from bulk density and
  closure in closed form over whole arrays (Clavier and Rust, 1976; new
  module `petrophys.interpretation.lithology.carbonate_minerals` with
  `solve_calcite_dolomite` and `solve_carbonate_minerals`): a whole well
  takes well under a second, where the per-sample optimiser needs about
  25 ms a sample. A sample whose bounded solution misses RHOB by more than
  0.02 g/cm3 is left unsolved and flagged. The new setting
  `carbonate_calibration` chooses the constants: `"generic"`, the default,
  takes clay from VCL at `rho_cl`, effective porosity, the pore fluid
  `rho_fl`, and calcite 2.71 and dolomite 2.87 g/cm3; `"costa"` is the COSTA
  field calibration (illite from GR, total porosity, a 1.19 g/cm3 brine and
  a 2.847 g/cm3 dolomite) and the "geolink" profile's default, which there keeps the
  per-sample solve.
- `LithologyModule(solver_options=...)` passes options such as
  `row_uncertainty` to its solver.

#### Permeability, saturation height and NMR

- `perm_method="wyllie_rose"` and `wyllie_rose()` follow the published
  Wyllie and Rose (1950) family k = P phi^Q / Swi^R (k in mD, phi and Swi as
  fractions), with presets in `WYLLIE_ROSE_PRESETS`: `morris_biggs_oil`
  (62,500, 6, 2; sqrt(k) = 250 phi^3/Swi, Morris and Biggs, 1967), the
  default; `morris_biggs_gas` (6,241, 6, 2); `tixier` (62,500, 6, 2; Tixier,
  1949); and `timur` (8,581, 4.4, 2; Timur, 1968). `p`, `q` and `r` override
  a preset. The old k = 250 phi^3/Swi^2 gave 108 mD at phi 0.30 and Swi
  0.25, where Morris and Biggs give 729 mD.
- `lith_vcl` permeability covers clean rock with VCL = 0, which it left
  without a value.
- `saturation_height_from_logs` converts heights in metres to feet for
  Pc = 0.433 (SGw - SGhc) h, where metric wells had their metres read as
  feet, takes a `tvd_curve`, and warns when it takes the depth index as true
  vertical depth; `fwl_depth` must be in the same unit and on the same
  datum.
- NMR: the clastic T2 cutoff default (`T2_CUTOFF_CLASTIC`) is 33 ms (Coates,
  Xiao and Prammer, 1999) instead of 32 ms, a bin centre, and
  `bvi_from_bins` and `ffi_from_bins`, also as `NMRData` methods, split the
  bin that straddles the cutoff by its log-T2 width
  (`t2_fraction_below_cutoff`). `partial_bins=False` keeps the whole-bin
  split; with `cutoff_ms=32.0` it matches the MBVI supplied with the
  Phillips data set.
- `calculate_hcpv_from_well` converts metric net pay to feet (metres were
  read as feet), weights porosity by thickness and SW by pore volume, and
  raises `ValueError` without NET_PAY, PHIE or SW instead of taking the
  whole interval as pay or porosity 0.15 and SW 0.30.

#### Net pay

- Unset cutoffs take each sample's lithology-family standard from
  `STANDARD_CUTOFFS`: clastic VCL < 0.40, PHIE > 0.08, SW < 0.60; carbonate
  VCL < 0.30, PHIE > 0.045, SW < 0.70. Limestone, chalk, dolomite and marl
  groups are carbonate, every other group clastic, and samples without a
  group take the family of the run's lithology. A cutoff set in the config
  applies to every sample. The trace records the cutoffs used
  (`cutoffs_by_family`, `cutoff_source`). The "geolink" profile uses VCL < 0.3 and
  SW < 0.7, with the carbonate PHIE cutoff only where a sample is labelled
  `"carbonate"`.
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
  context category. The eval package's categories had rules of their own: in
  the Aasgard Formation chalk of 15/9-2, one of its wells, the
  reservoir-quality labels of the porosity, permeability, hydrocarbon and
  lithology categories disagreed pairwise on 151 to 308 of 335 samples.
- Context labels change with the tables. Porosity and permeability grade
  clay volume as CLEAN_FORMATION (VCL < 0.25), MODERATE_CLAY and
  SHALY_FORMATION (VCL > 0.50). FZI takes the global hydraulic elements
  GHE_1 to GHE_10 of Corbett and Potter (2004), with BELOW_GHE_1 below
  0.0938 um. BVW is graded from VERY_LOW_BVW (below 0.03) to HIGH_BVW (0.10
  and above), classes that describe BVW rather than moveable hydrocarbons.
  Porosity, permeability, hydrocarbon and lithology grade reservoir quality
  from NON_RESERVOIR to EXCELLENT_RESERVOIR (below). A section that
  classifies one property gives a sample without a value its table's
  `NO_*_DATA` label, so a sample without MHI is NO_MHI_DATA rather than
  MINIMAL_MOVEABLE_HC. LOW_SW marks SW below 0.60, the WATER_BEARING edge,
  rather than below 0.50; the Rxo/Rt classes of the permeability context
  (0.8 and 1.2) read the workflow's INVASION_RATIO; and `mhi_confirmation`
  in the permeability metadata counts the samples of each MHI class.
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
  too. The eval package grades reservoir quality by a different rule in each
  category, and only its lithology category limits clay volume, on GOOD and
  EXCELLENT.
- In the Aasgard Formation chalk of 15/9-2 the four categories now give each
  of the 293 graded samples one grade: 254 NON_RESERVOIR, 16 FAIR_RESERVOIR,
  12 GOOD_RESERVOIR and 11 EXCELLENT_RESERVOIR, where porosity and
  permeability alone (the eval package's hydrocarbon rule) give 139, 28, 68
  and 58. Of the graded samples, 126 fall to NON_RESERVOIR on a clay volume
  of 0.30 or more and 11 rise to FAIR_RESERVOIR on the carbonate porosity
  limit; the 42 samples without permeability are not graded. On 30/6-5,
  25/5-1 and 15/9-14 the grades differ from those of porosity and
  permeability alone on 35.5%, 39.8% and 58.3% of the samples, mostly lower
  grades where the clay volume reaches a grade's limit, and PRIME_OIL_TARGET
  flags 276, 544 and 111 samples instead of 306, 1,947 and 240.
- Library functions: the classification enums of `petrophys.model` and their
  helpers classify through `CLASS_TABLES`. `classify_porosity` and
  `classify_sw` read values as fractions unless `percent=True`, instead of
  reading every value above 1 as percent, and warn when fractions look like
  percent (porosity above 1, SW above 2), so an unclipped SW of 1.02 is
  WATER_BEARING. `get_vcl_quality_code` gives 0 and `get_invasion_type` None
  for a value that is not finite; `is_clean_sand` and `is_shale` default to
  the VCL classes (`threshold=None`). `InvasionModule` counts its
  conductive, minimal and resistive percentages with the classes of its new
  `class_bands`, `CLASS_TABLES["RXO_RT"]` by default.

#### Context exports

- After a context export's value-range QC, PHI_NORM, RQI and FZI are
  recomputed from the porosity (PHIE_CC, then PHIE, then PHIT) and
  permeability (PERM_CC, then PERM) the QC left, and are empty wherever
  either input is missing, rejected samples included; PHI_Z takes the
  recomputed PHI_NORM, and BVW is empty where no porosity is left. Values
  computed from rejected samples therefore no longer reach the curve
  statistics, the derived-log list, DATA_COMPLETENESS or the category
  analyses: the category exporter recomputes these logs for each group right
  after its QC. Where the QC leaves a segment without porosity or
  permeability, the "geolink" profile keeps the values computed before it, and it
  keeps the workflow's PHI_Z.
- Depths and intervals carry the well's depth unit: `formation_info` and
  `lithology_group_info` give it as `depth_unit`, the bad-hole intervals of
  `BADHOLE_QC_CHECK` end with it, and the KH note gives KH in mD times the
  depth unit. The "geolink" profile writes them without a unit, as the eval package
  does.
- The bad-hole QC of a context (`BADHOLE_QC_CHECK` and the BAD_HOLE note)
  follows the rule of the bad-hole step, so it agrees with the BAD_HOLE
  curve, and counts the samples without a bit size (`no_bit_size_points`)
  and those neither test reaches (`not_assessed_points`).

### Breaking changes

- Context generation runs the same workflow as interpretation, with the
  calculation profile of its `workflow_config` (`"current"` unless set).
  Dataset names, vendor curves, and earlier context packages cannot select
  another workflow or provide archived numerical values.
- The workflow, `interpret`, `export_context`, `petrophys-context` and the
  MCP tools compute the `"current"` calculations by default: the
  calculations behind the GeoLink eval context benchmark with the
  corrections listed under "Corrected calculations".
  `calculation_profile="geolink"` computes the package's own. Under both
  profiles:
  - Defaults: GR endpoints picked from the log (`gr_min`/`gr_max` None:
    P5/P95) and a sandstone matrix in the clastic domain; with another
    `lithology_domain` the matrix is resolved within that domain.
    `standardize_input`, `vcl_cutoff` and `sw_cutoff` default to None and
    take their profile's value: input standardization is on under
    `"current"` and off under `"geolink"`, and unset net-pay cutoffs follow
    each sample's lithology family under `"current"` and are VCL < 0.3 and
    SW < 0.7 under `"geolink"`.
  - LAS header parameters override settings left at their defaults (for
    example `RHO_FL` over `rho_fl=1.0`); explicitly set values still win.
  - Without a measurement temperature, a given Rw is taken as the value at
    the median formation temperature and corrected along the temperature
    profile from there; the clean-wet Rwa fallback uses P5. When no Rw
    source is found, 0.05 ohm.m is used with a warning (`allow_default_rw`
    now defaults to True; set it to False to skip saturation instead).
    `"current"` also warns when it takes Rw at the median formation
    temperature or estimates it from Rwa.
  - PHIT and PHIE are always recalculated; interpreted input porosity is no
    longer kept as the active curve (no `PHIT_CALC`/`PHIE_CALC`).
  - The clastic lithology step needs RHOB, NPHI, PEF or DTC, VCL and PHIE.
  - Net-pay cutoffs are strict (`VCL < cutoff`, `PHIE > cutoff`,
    `SW < cutoff`). `NetPayModule` takes the carbonate standards only for
    samples labelled `"carbonate"`; the `"current"` workflow maps each
    lithology group to its family first.
  - Swirr stays as picked before permeability: the `permeability_finalize`
    and `rock_quality_finalize` steps are gone.
  - Bad-hole flags no longer fall back to pad calipers (HD1/HD2) when CALI
    is absent. `"geolink"` assumes an 8.5 in bit size when a well has no
    BS curve or header bit size and flags washout only with a caliper (DRHO
    alone does not flag it); `"current"` assumes no bit size, and where none
    is known flags washout from DRHO alone, also without a caliper (see
    "Bad hole").
  - `standardize_curves` gives core saturation curves the ordinary unit
    conversion (no percent inference); when two curves map to one standard
    mnemonic, the incoming curve takes the slot only with more valid samples
    and otherwise is dropped.
  - `SaturationAnalysis` no longer derives SO; `NumericalStatisticalAnalysis`
    counts 2- and 3-sigma deviations with `np.nanmean`/`np.nanstd` and exact
    thresholds.
  - Context packages use the eval package's layout: input curves are classified
    against the raw LAS files (`raw_input_dir`, default the sibling of a
    `*_processed` folder), value ranges use the eval package's rules (RMF from
    0.01 ohm.m; no SP, SO or NMR rules), BS gets statistics and SO does not.
    Their notes describe the calculations of the profile used, and under
    `"current"` their depths carry the well's depth unit.
- Library functions change with the corrections, for every caller:
  `wyllie_rose` takes `preset`, `p`, `q` and `r` in place of `a`;
  `SaturationModule` and `calculate_sw` no longer estimate Rwb, so Dual
  Water needs `rwb`; `total_porosity` and `PorosityModule` take the
  root-mean-square in gas crossover; `MultiMineralSolverV1` weights its
  equation rows; `bvi_from_bins` and `ffi_from_bins` default to 33 ms with
  partial bins; `saturation_height_from_logs` converts metric heights to
  feet; `calculate_hcpv_from_well` raises without NET_PAY, PHIE or SW;
  `NetPayModule` and the summaries sum midpoint cells; temperature helpers
  raise `ValueError` on an unknown unit label; `steps.net_pay` leaves
  non-reservoir samples out; `classify_porosity` and `classify_sw` read
  values as fractions unless `percent=True`; `get_vcl_quality_code` and
  `get_invasion_type` return 0 and None for a value that is not finite; and
  `steps.saturation` uses the published Indonesia equation and converts
  `cec` from meq/100 g. "Corrected calculations" describes each.
- `InvasionModule` reads the flushed-zone RXO first: its default `rxo_curve`
  is `"RXO"` (it was `"RSHA"`), and after `rxo_curve` it tries RXO, RSHA and
  RESS in that order, where it tried RXO, RESS and RSHA. `rxo_curve="RSHA"`
  keeps the former default order (RSHA, RXO, RESS).
- The diagnostics analyzers (`PorosityAnalysis`, `PermeabilityAnalysis`,
  `SaturationAnalysis`, `HydrocarbonAnalysis`, `LithologyAnalysis`) and
  `run_comprehensive_analysis` classify with `CLASS_TABLES` by default
  (`class_tables="current"`), so their labels and threshold texts change as
  "Classes and reservoir quality" lists; `class_tables="geolink"` keeps
  the eval package's. With the default tables, the reservoir-quality
  sections of porosity, permeability, hydrocarbon and lithology need clay
  volume (VCL, VCLAY or VSH) as well as porosity and permeability, and apply
  clastic limits unless `lithology_family` is given.

- The reservoir flag that picks the SWIRR_APP samples before permeability is
  recomputed from the run's own cutoffs. A `RESFLAG` or `NET_RES` curve already
  on the well, as an interpreted LAS carries, is no longer used for it, and a
  warning says so; set `reuse_input_resflag=True` to pick Swirr from the
  supplied flag as before. Re-interpreting an already-interpreted LAS therefore
  changes SWIRR_APP and permeability, which previously followed the older run's
  cutoffs silently.

- `DynamicWorkflowConfig` validates every setting when it is created and
  raises `ValueError` listing all invalid fields with the accepted values
  (previously, typos such as `hc_type="gass"` silently used defaults).
- `DynamicWorkflowConfig` uses `lithology_domain` to distinguish the
  clastic and carbonate mineral routes from per-depth lithology labels.
  `lithology_domain` defaults to `"clastic"` (set `"carbonate"` for
  carbonates, or `"auto"` to resolve it from the data); `matrix_type`
  defaults to `"sandstone"` in the clastic domain and to `"auto"` within
  any other domain.
- Tabular readers (`io.read_csv`, `read_csv_wells`, `read_ascii` without a
  units row, `Well.from_dataframe`) require `depth_unit` instead of assuming
  feet. `Well.depth_unit` accepts only metre or foot labels.
- The log viewer no longer searches parent directories for
  `meta_files/Litho_Nomenclature_Kingdom.xlsx`; pass `lithology_mapping` or
  set `PETROPHYS_LITHOLOGY_MAPPING`.

### Added

- Settings and functions behind the corrected calculations (see "Corrected
  calculations"):
  - settings `rmf_chlorides_mg_l`, `carbonate_calibration`,
    `sw_routing_vcl_cutoff`, `nd_clay`, the neutron-density clay point, and
    `lithology_curve`, the lithology code curve;
  - clay volume: `nd_shale_point`, `nd_clay_separation` and `ND_CLAY_POINT`
    in `petrophys.interpretation.clay_volume.nd_method`, `sp_polarity` in
    `petrophys.interpretation.clay_volume.sp_method`, and
    `VclModule(sp_fallback_only=..., nd_clay=..., nd_triangle_only=...)`;
    `nd_clean_line` accepts more matrix aliases (`ss`, `dolo`, `ls`, ...);
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
- Mineral solvers are versioned. `MultiMineralSolverV1` is the original solver as a public class:
  VCL and PHIE are fixed inputs, mineral volumes come from non-negative least squares over equation rows
  weighted by tool uncertainty (`row_uncertainty=None` solves them unscaled, as the original did), and
  samples whose mineral total misses `1 - VCL - PHIE` by more than 0.05 are left unsolved; its endpoints
  are frozen. `MultiMineralSolver` is version 2 (also exported as `MultiMineralSolverV2`) and is unchanged.
  `DynamicWorkflowConfig.mineral_solver_version` selects the clastic workflow solver (default `1`, the
  algorithm the workflow already used; it replaces the private `CanonicalMultiMineralSolver`), and
  `LithologyModule(solver_version=...)` selects it there (default `2`).
- Options behind the "geolink" profile, usable on their own: `InvasionModule(medium_deep_proxy=...,
  depthwise_curve_merge=...)`, `SaturationModule(sxo_quicklook_fallback="per_sample"|"whole_curve",
  depthwise_curve_merge=...)`, `PermeabilityModule(porosity_curves=...)` and
  `HydrocarbonAnalysis(porosity_curves=...)`. Defaults keep the current behaviour.
- `apply_value_range_qc`, `clean_curve_values` and `rule_for_curve` take a `rules` argument.
- `DynamicWorkflowConfig.swirr_flag_phi_cl` chooses the PHIE behind the reservoir flag that picks the
  SWIRR_APP samples. The default `"workflow"` keeps the reported PHIE; `"all_shale_phit"` recomputes PHIE
  for that flag only with clay porosity from the mean PHIT of all shale-lithology samples, as the eval
  package's main run computed PHIE. The trace records the rule and its phi_cl.
- Raw MDT pretest pressure picking through `read_mdt_pretest` and
  `pick_mdt_pressure`, configurable buildup QC, review flags, and a library
  plot of selected windows. `formation_pressures_from_mdt` passes accepted
  picks to station interpretation with explicit TVD metadata. Tutorial 15
  verifies the picker against 35 native C0020A LAS traces and their report.
- Unlabeled station-pressure contact inference through `infer_contacts`,
  exposing competing fluid hypotheses and unresolved contacts. The existing
  labeled FWL fit remains available.
- Public `build_context` / `export_context` and `petrophys-context` CLI with
  schema version 1.0, UTF-8 byte budgets, staged package publication and
  overwrite protection. The CLI always computes its context from input data.
- Standardized well-parameter and core-measurement readers and templates.
  Package generation accepts these alongside optional formation, lithology,
  and lithology-code tables.
- A carbonate mineral route: calcite and dolomite from bulk density and
  closure, with the COSTA field calibration (the published GR-illite
  relation) as `carbonate_calibration="costa"`.
- Context-package tops accept CSV/TSV/Excel, direct NPD/SODIR headers, explicit
  bottoms, depth units and rank selection without private default workbooks.
- `read_lithology_codes` validates codebooks and prefers Kingdom `Attribute`
  rock labels over a conflicting `Lithology Attribute` grouping, with a warning.
- A context API guide, data-quality notes and refreshed tutorial 19. CI now
  executes context generation and the real-file FMI tutorial.

- An MCP server (`petrophys-mcp`, `pip install "petrophys[mcp]"`) that lets
  LLM clients load wells, interpret them, inspect traces, summarize zones,
  plot and export results, with file access limited to chosen folders; a
  guide and tutorial 20 cover setup and a full session.
- `petrophys.workflow.config_schema()` and `workflow_steps()` describe every
  setting and step as data, and the documentation site serves `llms.txt`.
- Short public API: `petrophys.read`, `petrophys.plot` and the
  copy-by-default `petrophys.interpret`, with full docstrings;
  `DynamicWorkflowConfig`, `DynamicWorkflowResult` and `run_dynamic_workflow`
  are exported at the top level.
- `petrophys.DataQualityWarning` for implausible inputs: curves mostly outside
  their plausible range (with a unit hint such as "values look 1000x too
  large"), geology that does not overlap the logged interval, and
  unrecognized lithology labels.
- `DynamicWorkflowResult.to_las()` writes the interpreted well with the
  settings used (`PP_*` parameters) and step reasons; traces record the
  package version.
- Every skipped step records a specific reason, and `print_trace()` shows it.
- `formations_costa.csv` labels the logged interval of each Costa well as
  its single formation, KHARAIB-2, so formation summaries and context
  packages name the reservoir instead of treating the file as a whole well.
- `DynamicWorkflowResult.formation_summary()` tabulates gross, net and
  average properties per formation.
- NPD/GeoLink-style tops without a rank column get their stratigraphic rank
  from the name suffix (GP./Fm./Mb.), so mixed group/formation picks can be
  analyzed one level at a time with `run_formation_analysis(..., rank=...)`.
- Formations attached with `petrophys.read(..., formations=...)` calibrate
  core permeability per formation when they form a single level.
- Shared formation import for the model, viewer, and formation analysis, with
  explicit column, worksheet, well, depth-unit, reference and datum handling.
- External lithology precedence, numeric category metadata, and gap/unknown
  reports; core saturation fraction normalization with provenance.
- Input templates, an own-data guide and notebook, and fresh-kernel execution
  of the user-journey tutorials in CI.
- `config_schema()` entries carry the settings `group`. The documentation adds
  an interpretation-settings page generated from it, a page for the
  top-level `petrophys` namespace, a "Terms, curves and steps" reference and
  a review checklist for interpretations.
- `read_formations(..., rank=...)` keeps one stratigraphic level.
- MCP server, tested live with an OpenAI model on 18 realistic requests
  (all answered correctly; input tokens down from 599k to 268k and tool
  errors from 5 to 1 on the original 14):
  - new tools `check_data_quality`, `spectral_gamma`, `run_workflow` /
    `workflow_step_kinds` (custom workflows) and `interpret_wells` (many wells
    in one table);
  - rich tool descriptions and compact outputs; header values with units and
    notes (e.g. an Rw without its temperature); `interpret_well` returns its
    assumptions and review notes (whole-well totals include non-reservoir
    rock); `summarize_zones(rank=...)`; forgiving well and track names;
    `curve_values` is capped; paths are reported relative to the allowed
    folders.
- MCP deployment: `petrophys-mcp serve --http` (streamable HTTP or `--sse`)
  with bearer-token auth, required beyond localhost, and `/healthz`;
  `petrophys-mcp config <client>` prints the setup for Claude Code, Claude
  Desktop, Cursor, VS Code, Codex and OpenAI; `petrophys-mcp check` verifies
  folders and a live session; a Dockerfile and docker-compose.yml;
  `examples/mcp_openai_agent.py`; and `scripts/mcp_live_eval.py` for
  re-running the live evaluation.
- Custom workflows: `pp.Workflow` and `pp.workflow_step` build reusable
  interpretation recipes from built-in steps (`petrophys.workflow.steps`,
  including `pp.interpret` as one step) and your own functions. Inputs are
  checked before running; runs keep a full trace and export to LAS/JSON;
  `run_many` runs a field; workflows save and load as JSON. New guide page and
  tutorial 21.
- Tools promoted from the tutorials into the library:
  `find_core_depth_shift` / `shift_core_depths` (core-to-log depth matching,
  with per-core-run scans); `find_held_readings` / `held_reading_mask` (held or
  padded tool readings); `summarize_pay` / `PaySummary` (net pay, NTG, pay
  averages and hydrocarbon pore thickness for a window); the new
  `petrophys.evaluation.sensitivity` module with `cutoff_sensitivity`,
  `parameter_sensitivity` (tornado data) and `compare_saturation_models`;
  `sp_rw_in_window` (SP-derived Rw in a chosen window at a reference
  temperature); `classify_th_u_environment`, `thorium_clay_index` and
  `uranium_stripped_gamma_ray` for spectral gamma; `label_depths` and
  `flag_intervals`. Tutorials 06, 11, 12, CS3, CS5 and the complete tour call
  them instead of inline helpers.
- The API reference is reorganized: a landing page with one card per area
  (front door, data model and I/O, preprocessing, interpretation, evaluation,
  workflows, visualization, reporting and LLM agents, legacy), a page per
  section with grouped summary tables, and one page per function and class,
  each documented once with its other import paths noted. It is generated
  from each package's `__all__`, so it follows the code; old reference URLs
  redirect to the new pages.
- The user guide is rewritten as executed pages: every example on the
  quickstart, own-data, input-template, data-model, glossary, architecture,
  workflow, review, context, MCP, visualization, specialist-data and
  capabilities pages runs at each documentation build, so the outputs and
  figures shown are always current. "What Petrophys can do" is a gallery of
  every capability with the call that produces it.
- A complete-tour notebook covering the whole library on four bundled GeoLink
  wells, and a real spectral-gamma geochemistry part plus a second-well core
  calibration in tutorial 12.
- `scripts/check_notebook_outputs.py`, run in CI, rejects tutorial notebooks
  with unexecuted cells, errors or local paths in their saved outputs.

### Fixed

- Named carbonate and marl cutoffs now agree with matrix routing. Marl remains
  an explicitly warned routing convention, not a calibrated mixed-rock model.
- The bundled 30/6-5 formation CSV declares metre depths and group/formation
  hierarchy. Direct DataFrame and file codebooks use the same validation.
- Netlify uses the strict documentation build and its required dependencies.
- Log-plot warnings no longer repeat a depth-reference note reported by both
  the formation and the lithology import.
- `workflow_steps()` descriptions say which steps `pp.interpret` does not run.
- Net-pay trace decisions report thicknesses in the well's depth unit; feet
  wells (e.g. the COSTA set) were labelled "m". The core-on-logs plot title
  uses the well's unit too.
- `lithology_domain` now keeps `matrix_type="auto"` inside its family:
  "carbonate" resolves to limestone or dolomite (the evidence picks between
  them) and "clastic", the default, to sandstone. When the matrix evidence
  points elsewhere, the domain wins and a warning names the conflict (on a
  default run it suggests `lithology_domain="carbonate"`). With
  `lithology_domain="auto"` the evidence decides both.
- Warnings escalated to errors (for example
  `warnings.simplefilter("error", pp.DataQualityWarning)`) now stop
  `pp.interpret` instead of being recorded as a failed step.
- An invalid `lithology_domain` names that setting and suggests the closest
  domain; `run_selected_steps` records skip reasons under the same step names
  as a full run.
- Interpreted LAS files record `PP_RW` and its temperature when
  `rw_meas_temp_f` is given (it was dropped), and a lithology-dependent Archie
  `a` is written as "per lithology".
- `CoreData.filter_by_depth` slices only per-plug arrays in its metadata;
  strings and dictionaries of the same length made it fail.
- `TemperatureProfile.from_bht` reads "degC"-style BHT labels as Celsius
  (only a bare "C" was converted).
- Buckles and Hingle plot labels stay inside the frame without overprinting;
  Buckles colours permeability on a log scale; Pickett plots no longer
  overlap minor tick labels.
- `create_heatmap` centres each cell on its coordinate (cells were shifted by
  half a cell), draws unevenly and log-spaced bins such as NMR T2 correctly,
  and also accepts cell edges.
- Analytics charts no longer flip the depth axis back when drawn twice on the
  same axes.
- `core_data_from_well` returns core water and oil saturation as fractions,
  converting percentages labelled as fractions with a warning, as its
  documentation states. The conversion rule now lives in
  `petrophys.model.units.core_saturation_to_fraction`.
- `IntervalAnalyzer` gives the dominant lithology of an interval as the
  label of the selected lithology intervals, else as the lithology group of
  the codes on the well's lithology code curve (such as "sandstone"); it
  gave lithology intervals as category numbers and was empty on wells with
  `LITHOLOGY_GEOLINK` codes.
- The core plotting helpers (`plot_calibration`, `plot_poro_perm`,
  `plot_core_on_logs`) and `GEOLINK_LITHOLOGY_CODE_LABELS` are exported from
  `petrophys.visualization` and appear in the API reference.
- The DLIS readers name the `petrophys[dlis]` extra when dlisio is missing,
  and `result.summary()` rounds net thicknesses to 0.01.

- A workflow reused across wells no longer carries one well's resolved
  matrix into the next.
- Celsius temperatures labelled like "deg C (from log)" are converted wherever
  header temperatures are read, and in temperature curves under `"current"`;
  units are parsed by one shared implementation.
- Reference reports label RMF methods as RMF (they were labelled as RW).
- Vendor mnemonics and units (e.g. `GR_EDTC`, `RHOZ` in kg/m3) plot correctly
  before interpretation.
- `run_formation_analysis` refuses to summarize a failed interpretation.
- LAS files are decoded as UTF-8 first, and unreadable files raise a clear
  `ValueError`.
- The ML extras install on Python 3.9 and 3.10 (the ML stack itself needs
  3.11+), and incompatible scikit-learn versions raise
  `ModelCompatibilityError`.
- The MCP server reports each range-screening warning once.
- The S_V diagnostic is labelled as the overburden stress increment from the
  top of the analysed interval, not the overburden from surface; under
  `"current"` the context notes say so.
- Docstrings corrected: the closed form of the Coates calibration, the units
  of the Kozeny-Carman constant, the Waxman-Smits first-order note,
  `sorting_coefficient`, the `arps_equation` example, the North Sea
  equation, the strict inequalities of the net-pay helpers and the
  `cutoff_sensitivity` example; `dual_water` no longer suggests
  Rwb = 0.3 Rw.

### Changed

- Formation model and summary loaders no longer add a final bottom 100 units
  below the last top; supply `terminal_depth` for top-only analysis, or use
  `pp.read_formations` for point annotations.
- Invalid geology rows raise instead of being skipped silently, summary
  totals count the covered depth union rather than the sum of nested rows,
  and unknown reader or workflow options raise instead of being ignored.
- `load_lithology_csv` is kept for compatibility; new code should use the
  shared validated readers.
- Release steps and code-placement rules moved from the documentation site to
  `CONTRIBUTING.md`; the Examples page is merged into Tutorials.
- Public modules are organized under `model`, `io`, `preprocessing`,
  `interpretation`, `evaluation`, `workflow`, `reporting`, and `visualization`.
  Consult the API reference when migrating imports from 2.x.
- The bundled imputation pack predicts DTC, NPHI, and RHOB.
  Estimators must have held-out-well R2 of at least 0.5. Other measured curves
  remain usable as companions. Imputation fills existing null samples and
  records provenance in LAS exports.
- Mineral inversion reports non-resolvable and zero-volume failures as missing
  values with QC flags. A reduced calcite-dolomite model is available when
  carbonate evidence supports it. Workflow traces record mineral selection.
- Context packages are computed by the same workflow as direct
  interpretations; no archived context payload or benchmark manifest is read.
- Package checks install the wheel and execute a sample workflow. Documentation
  builds treat warnings as errors, and release tags must match package versions.
- Third-party data provenance and model validation records accompany the package.

### Known limitations

- A whole-well run uses one Rw, corrected for temperature at a constant
  salinity, although shallow formation waters are usually fresher, and the
  clay indicators disagree in undercompacted mudstones. Default runs can
  therefore still flag marginal pay (SW about 0.5) in shallow Tertiary
  sections that no lithology code log covers: on 25/5-1, 259 m of the
  310.4 m lies above 2984 m, nearly all of it uncoded. Read the results per
  formation (attach tops and use `result.formation_summary()`), or give the
  shallow interval its own Rw (a zoned header parameter such as `RW_Z1`
  between `TOP_Z1` and `BOT_Z1`, which
  `petrophys.workflow.run_dynamic_workflow_with_zones` applies). A
  depth-varying Rw is future work.
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
