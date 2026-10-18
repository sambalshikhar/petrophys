# Open scientific-method questions

Raised during the October 2026 release audit. Every item below was decided on 17-18 October 2026; see METHOD_DECISIONS.md for the decision taken (implemented, deferred as not affecting results, or left to the maintainer). Clear defects were fixed separately (see FINDINGS.md).

## core

- _log_at_depth treats a gap as more than 1.5 median log sample steps between finite samples, so a plug next to even a single NaN sample gets NaN. Is that tolerance acceptable, or would you prefer a looser limit (e.g. a fixed max bridging distance in metres)?
- correct_porosity_with_grain_density: beyond the first/last GDEN plug inside the cored interval, it now uses default_rho_ma, as the docstring says. Would you rather hold the nearest plug's grain density there?
- CoreMechanical.scale_average keeps the old rounding of window_m to an odd number of samples (half = round(window/dz/2)) rather than an exact +/- window_m/2 depth window. Should it use the exact depth window instead?

## dynamic workflow

- phi_cl=0.15 and rmf_meas_temp_f/rm_meas_temp_f=75 given explicitly count as 'left at the default', because the dataclass cannot tell explicit from default. Should these defaults become None sentinels (an API change), or is the documented behaviour (now in the docstrings) acceptable?
- Rmf precedence: should an explicit config rm / rmf_salinity_ppm / rmf_chlorides_mg_l take priority over header RMF (and rm over header CLFL), as config rw does over header RW? For now only a warning is added.
- PERM_CC from the core poro-perm transform is applied over the whole well: it reads higher in the Heather shale than in the Etive, and hits the 10,000 mD clip in the shaly/coaly Ness. Should it be limited to the cored interval or rock type, or flagged where it extrapolates?
- apply_hc_correction only writes NPHI_HC/RHOB_HC (now documented). Should porosity actually use the corrected curves when the option is on?
- The evaporite lithology_domain runs the clastic solver and cutoff family, and labels uncoded samples EVAPORITE. Is a dedicated evaporite routing intended?
- pp.interpret applies a zone-only RW_Z1 (via _constant_zoned_parameter) to the whole well. Should zone-only parameters apply only inside their zone, or give a warning? This is api/zoned-runner behaviour, not changed here.
- Should implausible numeric settings (surface_temp_f=-500, bht_f below the surface temperature, nphi_ma=5, rho_ma=26.5, m=20, rw=1e6) be rejected by _validate_workflow_config? Which ranges?
- Should a measured TEMP log be used as the temperature source or BHT anchor instead of the modelled gradient? It is now kept as TEMP_LOG with a warning.

## elec perm temp sat

- #20 Should RwModule's rxo_rt method prefer RXO (flushed zone) over RSHA? It currently reads ['RSHA','RXO','RESS'].
- #48 Should perm_fallback_method='timur'/'lith_vcl' be tried before coates? _auto_empirical_methods always puts coates first, and tests/test_permeability_lith_vcl.py asserts that order, so I left it.
- #1 Should VclModule detect reversed SP polarity (sp_polarity) when it picks SP baselines automatically?
- #2 Should VclModule's default GR baselines be P5/P95 instead of the log min/max? (docstring now documents the current behaviour)
- #6 Is the wet-zone Rwt approximation in dual_water acceptable when SaturationModule routes shaly carbonates to it, or should it use the Clavier-Coates-Dumanoir Swb/Swt form?
- #7 Should the sonic detectability gate include K_dry frame uncertainty?
- #10/#11 When no VCL curve exists, should PorosityModule and SaturationModule warn, or return NaN PHIE / label the model, instead of silently using VCL=0?
- #46 Should north_sea_shaly_sand return NaN instead of Sw=0 where the shale term exceeds 1/Rt?
- #35 Should RmfModule.calculate_from_well read the RMFT/MFST header parameters for the Rmf measurement temperature?
- #24 Should rw_from_ssp flag or stop clipping valid results outside [0.01, 10] ohm.m?
- #44 Should _build_same_well_core_prediction default to include_global=False (latent overwrite of formation PHIE_CC for porosity-only cores)?
- #49 Should peer-well core calibration run when the target well has no core porosity, and should rejected peers be reported?

## evaluation

- NumericalStatisticalAnalysis SH_MIN: Eaton is applied to an S_V increment from the top of the log together with an absolute hydrostatic PP, which gives SH_MIN < PP. Should SH_MIN need an absolute Sv (an overburden from surface, or an input), or should it not be published?
- PP_HYDROSTATIC adds a fixed 100 m seawater column whenever the deepest sample is below 200 m. Should water depth and datum be inputs, with no seawater term by default?
- S_V integrates only finite-RHOB samples, so a RHOB gap adds zero overburden. Should S_V be NaN below an internal gap, or should the density be interpolated across it?
- The trend-zone threshold |gradient| > 0.1 is per well depth unit, so metre and feet wells classify differently. Should the gradient be converted to per-metre first?
- Zone/formation HCPV and BVW are products of unweighted sample means (avg_phie*(1-avg_sw)*net_pay), not sum(phi*(1-Sw)*h) with thickness-weighted averages as in summarize_pay and calculate_hcpv_from_well. Should zone_summary switch to the summed and thickness-weighted forms?
- calculate_hcpv_from_well, calculate_zone_summary, calculate_pay_averages and BvwModule read PHIE/PERM, while the interpretation's SW/NET_PAY/BVW use PHIE_CC; BvwModule.add_to_well also overwrites the workflow BVW. Should they all prefer PHIE_CC/PERM_CC?
- NetPayModule/calculate_net_pay apply carbonate cutoffs only to the exact label 'carbonate' (as documented), while cutoff_family maps limestone/chalk/dolomite/marl to carbonate. Should NetPayModule use cutoff_family?
- calculate_hcpv_from_well: should net_pay_ft count only pay samples with finite PHIE/SW (so the documented identity holds), instead of the coverage warning added now?
- SaturationAnalysis gas section: GAS_CORRECTION_EFFECT cannot be reached by construction (PHIT_GAS_CORR only exists where the ND branch already labelled the sample). Should the rule be made reachable, or removed from the threshold text?

## io

- Teapot LAS: should CORR be an alias of DRHO, CALD of CALI, and should header BS1 be used as bit size? These alias changes would alter standardisation, and therefore the bad_hole step, for any well carrying those mnemonics, including possibly the benchmark wells. Not changed.
- LAS export uses fixed 5 decimals, so PERM values below 5e-6 are written as 0. This is known to be intentional for benchmark output; do you want significant-figure formatting outside benchmark reproduction?
- Formation tops without a depth_unit: should this be an error (like CSV well logs), rather than an assumption of the well unit with the new warning?
- Should non-LAS inputs (DataFrame, CSV, DLIS) get the duplicate and non-monotonic depth warning that the LAS reader gives?

## lith imaging

- resolve_from_core_matrix: nearest-of-2.65/2.71/2.87 voting with no minimum confidence returns carbonate/limestone (conf 0.54, votes 66/62/15) for 30/6-5, whose core GDEN median is 2.66, and dynamic_workflow applies it without a confidence gate under lithology_domain='auto'. Should this use a median rho_ma, a confidence gate (as in LithologyModule's 0.8), or core GDEN when present? (Reproduced; not changed.)
- _CORE_POROSITY_CURVES includes PHIE_CC, but datasets/costa/README says COSTA PHIE_CC is the renamed upstream log PHIE, not core. That makes HW-3 resolve to a dolomite matrix. Should PHIE_CC be dropped from the core-porosity sources, or should COSTA data be excluded? Any change would need a benchmark gate.
- MultiMineralSolver / MINERAL_PROPERTIES / nd_crossplot_data use grain density rather than the density-log response (MatrixProperties.log_density) for evaporites. Pure halite at RHOB 2.032 solves to 3.6 p.u. porosity. Should the RHOB response rows use log_density? (Reproduced; this changes the equation, so not changed.)
- identify_lithology_zone uses fixed RHOB/NPHI boxes that misclassify clean water-filled lithology lines (20% limestone comes out 'mixed', 20% dolomite 'shale'). Should it be rewritten against the N-D lithology lines, or deprecated? It is unused and not exported.
- _collect_flattened_curve_groups default (no prefixes) generic pass groups any numbered curves (e.g. AT10/AT20/AT90) as an image when no FMI prefixes match. Should the generic fallback be kept?

## mcp cli pkg

- HTTP server state: should each MCP session get its own Session (isolated wells/interpretations) instead of one process-wide Session? This patch only documents the sharing and blocks silent overwrite of a well id.
- Resource limits: should load_well have a maximum file size, and should the session cap or let clients unload stored wells and interpretations (an unload tool) on long-running HTTP servers?
- check_data_quality bad hole now uses the current-profile rule (caliper test where a bit size is known, |DRHO| alone elsewhere, after unit standardization and range screening), so it matches interpret_well. Is that the intended basis for the report, or should DRHO-only wells be reported as 'not assessable'?
- NTG in zone tables is net pay / gross (library ZoneSummary definition); conventional NTG is net reservoir / gross. Keep the definition (now stated in averages_basis) or change it?
- ml extras are capped at scikit-learn<1.9, joblib<1.6 and xgboost<3.3: is the reason pickled-model compatibility, and should that reason be written in pyproject?

## misc

- harmonic_mean/geometric_mean silently discard zero and negative values, so harmonic_mean([10, 0]) = 10, whereas scipy.stats.hmean gives 0 (a zero-k barrier). Should zeros make the harmonic mean 0, or should the exclusion be documented?
- Cased hole: should porosity be masked, or flagged, above the casing shoe from the header casing depth (e.g. CBL1 502 ft on 49025107090000_238176)?
- settings_used: should it record the cutoffs the calculation profile resolves (vcl/phie/sw under geolink) rather than None? I changed only the doc, because changing summary output could alter context packages.
- calculate_porosity rhob_fluid guard: are the accepted bounds of (0, 2] g/cm3 right, or would you prefer a tighter range of about 0.5-1.5 or a warning instead of a ValueError?
- With rename_input_curves=True, standardize_curves resolves AT60/AT90 collisions by its own rules. Should it also prefer AT90 for RDEP? I left it unchanged.

## model

- FluidProperties gas hydrogen_index=0.3 and saltwater HI=1.0 are inconsistent with the table's own HI=9n*rho/(12+n) at the tabulated densities (about 0.45 for methane at 0.2 g/cc, about 0.95 for 1.1 g/cc brine). Should the tabulated values be changed (DefaultParameters.GAS_HYDROGEN_INDEX=0.3 is separate), or documented as conventional defaults?
- ZoneSet.get_at_depth gives a shared boundary to the upper zone (Zone.contains includes the bottom), while label_depths uses top <= depth < bottom. Should log_data Zone/ZoneSet follow the top-inclusive, bottom-exclusive convention?
- Curve arithmetic with a scalar keeps the left unit, so the docstring PHID = (2.65 - RHOB)/1.65 is still labelled g/cc unless the user sets the unit (the docstring now does this). Is that the intended contract, or should derived curves always have an empty unit?

## nmr geochem

- calculate_elastic_properties: should Poisson's ratio stay clipped to [0, 0.49] while E/K come from the unclipped velocities, and K/E stay clipped to 0, or should physically impossible values become NaN (or E be made consistent with the reported nu)? Should the DTS upper window of 600 us/ft be widened for soft shallow sediments? (only documented for now)
- XRFData.mineral_proxies siderite index: the S > 0.5 pyrite penalty assumes wt%, but interpret_elemental_spectroscopy passes dry-weight fractions, so ELEM_SIDERITE_INDEX never applies the penalty. Should the threshold follow the units (0.005 for fractions)? This changes workflow output, so it would need a benchmark gate.
- bin_cumulative: one NaN bin makes the rest of its row NaN. Should NaN bins be skipped (nancumsum) or left NaN as now?

## reporting quality

- #4 _load_excel_formation_segments: when any pick of a well has an ' FM.'/' GP.'/' MB.' suffix, all other picks are dropped without a warning. Is this filter intended (NPD tops), and should it warn or be documented in the exporters?
- #17 value_ranges: DCAL is aliased to the absolute-caliper rule (4-40 in), but in Schlumberger usage DCAL is the differential caliper (CALI-BS). This removes 96% of DCAL on 16_7-1. Should DCAL get its own rule (about -5 to +5 in) or none?
- #17/#18 apply_value_range_qc/clean_curve_values do not read curve units. Should they skip or convert curves in non-standard units (NPHI in p.u., CALI in mm, DT in us/m) instead of only documenting it? Also, should +/-inf under 'clip' rules (SW) become NaN or be clipped?
- #19 WellboreQC: the roughness threshold 0.5 is applied to dCALI/dDepth, so the result depends on the depth unit (about 3.28x between m and ft). Which depth unit is the threshold meant for?
- #23 WellboreQC washout_threshold (1.5 in) is stored but never used, so a smooth overgauge hole is never flagged. Should it add a CALI-BS > threshold test, or should the parameter be removed?
- #8 Should lithology_mapping_file also accept the read_lithology_codes CSV/template format? Changing the reader would change GeoLink/Danish group names, so only the docstring and duplicate/fractional guards were changed.
- #20/#21/#22 COSTA: _carbonate_volumes sets VCALC=1, VDOLO=0, LITHOLOGY_GEOLINK=11 where reference rows and RHOB/PHIE/VCL are missing. The notes and trace also say 'PHIE_CC values are from cores' and give a PHIE_CC-only CROSSOVER_FLAG basis. This code runs only to reproduce the COSTA benchmark package. Should the package be regenerated with the disclosure or with NaN volumes?

## units std

- RDEP alias ranking: ALIAS_MAP['RDEP'] lists AT60 before AT90, so with equal sample counts AT60 becomes RDEP/Rt in both standardize_curves and add_standard_curve_aliases. Should AT90 (the deepest AIT curve) rank first? Changing the order affects benchmark alias resolution.
- HCGR (HNGS uranium-free computed GR) is in ALIAS_MAP['GR'], so get_standard_mnemonic('HCGR') returns 'GR'. Should it map to CGR? get_aliases('GR') also lists SGR, CGR and HSGR, which reverse-map to other standards.
- FT-SWC-UNITS: standardize_curves deliberately leaves percent core SWC/SOC labelled 'frac' unconverted; tests/test_benchmark_behaviour.py::test_core_saturation_gets_the_ordinary_unit_conversion requires this outside benchmark mode, and only the core calibration and the plots convert. Should standardize_curves and the pp.interpret output well convert core saturation to v/v, as complete_tour.ipynb cells 20 and 55, figs 033/036 and the docs wording suggest? If not, the notebook text and figures need correcting instead.

## viz

- #1/#42: create_density_neutron_crossplot puts the sandstone matrix point at NPHI 0.0, but lithology_crossplots._MATRIX uses -0.02 (CP-1 shows about -0.02 to -0.035). Should the viewer use -0.02 to match?
- #49: plot_poro_perm draws the fitted transform over phi 0.001-0.40, past the plug range (up to about 1.5e6 mD). Should the line stop at the data's porosity range, or show the extrapolated part differently?
- #46/#51: N-D (NPHI -0.05..0.45) and MID (35-65 us/ft, 2.5-3.0 g/cc) axis limits are still fixed. A note now gives the off-chart count. Do you want xlim/ylim arguments, or limits that follow the data?
- #45/#50: the log viewer header shows raw NPHI under the PHIN alias, and header text overlaps in multi-curve tracks. Should the header show the source mnemonic, and should header layout spacing be changed?
- #43: the CS2 notebook (cell 34) should pass data_label='SW_INPUT (reference)' to create_saturation_height_plot. Should I edit and re-execute the notebook, or leave it to you?

## workflow other

- Issue 2: should custom-step parameters be passed to the function in their JSON-converted form (list/float/str) on every run, so the first run and a reloaded run behave the same, or should the docs just state that tuples, numpy scalars and Paths come back as list/float/str after save/load/with_params?
- Issue 6: with matrix_type='limestone' and the default matrix_dt=55.5, PHIS uses the 47.6 table value while PHID uses config.matrix_density=2.65. Should explicit WorkflowConfig endpoints always win, or should matrix_dt/matrix_density default to None (meaning 'from matrix_type')? I only apply matrix_dt/fluid_dt when they differ from the defaults. config.rho_cl is still unused.
- Issue 7/11: should the workflow keep input PHIE_CC/PERM_CC (for example renamed PHIE_CC_INPUT) instead of removing them before permeability? For now the removal is only warned about.
- Issue 17: the quick-look pay-zone thresholds (GR<60 API, Rt>10 ohm.m, independent of gr_clean/gr_shale), the 2.0 minimum thickness in the well's depth unit, and thickness = depth[stop-1]-depth[start] (one sample short): should these become documented, configurable settings in a fixed unit?
- Issue 19: LITH/LITHOLOGY header text such as 'sandstone' is accepted as lithology_domain (validated as a lithology with Archie coefficients). Is that the intended mapping, or should it be restricted to the clastic/carbonate/auto domains?
- Issue 20: the fallback audit reason still joins every warning of the file. Should a missing curve with no step-specific reason report only its own step's messages (or nothing)?
- Issue 5: the SP-derived Rw is the median of per-sample Rw at the median TEMP and is then Arps-shifted per depth. Is that the intended basis, or should the per-sample SP Rw be used directly?

# Clear defects left unfixed

Each item states why it was not fixed in this round.

## dynamic workflow

- DynamicWorkflowConfig.__post_init__ still mirrors lithology_domain into the deprecated alias, so a direct dataclasses.replace(cfg, lithology_domain=X) without lithology=None still hits the conflict. The public run_dynamic_workflow path is fixed; changing the alias semantics in __post_init__ would change how mixed alias/domain inputs are resolved, so it was left for the maintainer.
- phi_cl / *_meas_temp_f sentinels: only the docstrings were changed; see method_questions.
- PERM_CC extrapolation, evaporite routing, zone-only RW_Z1, numeric range validation: methodological or outside this module's minimal scope; see method_questions.

## elec perm temp sat

- The changes are not committed and the patch file is not written; both still need doing in Q:/fixwt/elec_perm_temp_sat.
- Whether the existing module tests, test_context_profiles.py and test_benchmark_behaviour.py pass is not known yet (logs t1.log/t2.log under Q:/audit2/work_elec).
- #45 Simandoux phi=0 -> Sw=0 did not reproduce: the base already has the gated stable form, and it returns 0.625.
- #49 peer-well core calibration not fixed: it changes workflow step gating and is ambiguous.
- #44 latent include_global default not changed: no public result is affected.

## evaluation

- The root cause of Sw=0 at Rt=inf in the shaly-sand models is in petrophys.interpretation.saturation (other shard); only the compare_saturation_models output is masked here.
- Geomechanics (SH_MIN, seawater term, S_V gaps, gradient threshold), HCPV averaging, PHIE vs PHIE_CC, NetPayModule carbonate family and the GAS_CORRECTION_EFFECT text are left as method questions.

## io

- export_las: string parameters containing ':' (e.g. 'ratio 1:2 used') still re-read truncated, because lasio splits ~P lines at the first colon. Fixing this would mean changing the value text (escaping) or the reader, which is not a clear-cut fix.
- read_csv (single well): a numeric UWI column still loses leading zeros in the derived well name ('0012' becomes '12'). Reading that column as text would turn it from a numeric curve into a skipped text column and change the output curves.
- export_las performance (lasio formats each cell in Python): this is a performance item, not a defect fix; left as is.

## lith imaging

- The resolve_from_core_matrix result for 30/6-5, the PHIE_CC-as-core-porosity issue for HW-3, the evaporite log-density endpoints and identify_lithology_zone are scientific-method questions; they are listed under method_questions.
- Generic numbered-curve grouping when no prefixes are given (AT10/AT20/AT90 read as an image) is a heuristic choice; listed under method_questions.
- Pre-existing failures in test_correctness_fixes (2) and test_critical_findings_regressions (1) are outside this group.

## mcp cli pkg

- RP-EPSLOG: the EpsLog redistribution permission is a licensing and evidence issue, not a code defect. The maintainer has to add a written permission file.
- RP-LICEXPR: License-Expression 'MIT' versus the bundled CC BY, CC BY-SA and unlicensed data. This is a packaging and legal decision for the maintainer.
- RP-URLS / docs_beginner install and README 404s: the repository is private and the package is not on PyPI. The URLs can only be fixed once a public location exists.
- RP-README-REL: the relative README links need absolute target URLs, which do not exist yet (same cause as RP-URLS).
- RP-PINS: the version caps on the ml extras are probably needed for the pickled model; listed as a method question.
- SEC:http-shared-state per-session isolation: this is an architecture change. It is documented and the silent overwrite is blocked, but isolation itself is left to the maintainer.
- SEC:resource-limits: needs policy choices (size limits, eviction); listed as a method question.

## misc

- RP-GEOLINK, RP-REALPORE, RP-NORSKHYDRO, RP-KGSMETA: licensing and provenance decisions for the maintainer, not code defects. Dropping or relabelling bundled data would also change the benchmark inputs.
- RP-CHANGELOG: the release date is the maintainer's decision.
- undocumented-exports (docs/api/_generate.py AREAS): docs coverage gap, not a behaviour defect. Adding sections needs a decision on where they go on the API pages.
- verify_epslog_data.py / verify_phillips_data.py overwrite --report silently: overwriting an explicitly named output path is normal CLI behaviour, so not a clear defect.
- Pre-existing failures in test_critical_findings_regressions, test_correctness_fixes (x2) and test_mcp_deploy, present on base 8b8e3c4 and outside this group.

## model

- Bands.labels can still collapse probes for edges of about 1e17 or more (the +/-1.0 offsets). Reviewer noted it as a limitation; no canonical table is affected, so it was left alone.
- Well.from_dataframe with an unnamed non-Range index and no depth column still ignores the index. The index is used only when its name matches depth_column, because any other choice would be a guess.

## nmr geochem

- XRF siderite S>0.5 threshold is unit-dependent: left because the threshold is a method choice and fixing it changes workflow output (see method_questions)
- Elastic Poisson clip inconsistency: only documented; changing the clip is a method choice (see method_questions)
- Pre-existing failure in test_calibrated_dlis_alias_beats_uncalibrated_native_curve: belongs to the DLIS reader, outside this group

## reporting quality

- #20/#21/#22 COSTA VCALC=1 default and the PHIE_CC provenance/crossover note text: this code runs only to reproduce the COSTA benchmark package, so any change would break the byte-for-byte reproduction. Raised as a method question.
- #11 (part) Windows reserved names (CON, NUL) in formation names are still not handled; minor, not reproduced.
- #32/#33 performance (COSTA export I/O, import time): not defects in scope.
- Pre-existing failure test_calibrated_dlis_alias_beats_uncalibrated_native_curve (curve_standardizer) is outside this group.

## units std

- RP-MODEL: the GeoLink-derived pretrained imputation model (.joblib) is shipped with unresolved rights. This is a licensing and packaging decision for the maintainer, not a code defect.
- FT-SWC-UNITS: complete_tour.ipynb prose (cell 15 claims two UserWarnings; cell 20 prints a conversion), figs 033/036 with no core Sw points, and the docs wording were not edited. Whether the standardizer should convert core saturation is listed under method_questions. The notebooks would also need re-execution.
- Three existing tests (test_correctness_fixes collision tests and test_critical_findings_regressions::test_calibrated_dlis_alias_beats_uncalibrated_native_curve) fail on base 8b8e3c4 itself, before this patch. They pin the old duplicate-curve behaviour and are outside this group's scope.

## viz

- #48 (part): plot_poro_perm's colour bar is still 'Depth' with no unit (CoreData has no depth unit), and NMR panel / multiwell overlay axis units were not changed (not in the files I reproduced against).
- #45/#50: header overlap and the PHIN alias are cosmetic layout choices, listed under method_questions.
- Notebooks were not edited (e.g. CS2 data_label) because they hold executed outputs.

## workflow other

- Issue 2 (custom-step parameter types after save/load): this is an API-contract decision, so it is listed under method_questions.
- Issue 17 (quick-look pay-zone thresholds, minimum-thickness unit, thickness definition): these are method choices, listed under method_questions.
- Issue 19, the lithology part only: header LITH='sandstone' is still accepted as lithology_domain (MUD text is fixed). Listed under method_questions.
- Issue 20, the second half: the fallback audit reason still joins all warnings. Only the PHIE_CC reason text was corrected.
