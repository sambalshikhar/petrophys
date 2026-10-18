# Decisions on the open method questions

Decisions taken on 17 October 2026 for the items in METHOD_QUESTIONS.md.
Only items that change results (wrong numbers, ignored settings, lost data)
are implemented; cosmetic, policy and architecture items are deferred.
Every change keeps the benchmark context packages byte-identical: where a
decision alters a calculation the GeoLink/COSTA/Danish profiles use, the old
behaviour is kept inside `petrophys._internal.benchmark.historical_benchmark()`.

## Implemented (affect results)

### Workflow (dynamic_workflow.py, standard_workflow.py)

- Explicit config `rm`, `rmf_salinity_ppm`, `rmf_chlorides_mg_l` win over
  header RMF/CLFL, as `rw` does over header RW.
- `phi_cl`, `rmf_meas_temp_f`, `rm_meas_temp_f` default to `None` sentinels
  (resolved to 0.15 / 75 / 75), so an explicit value is never ignored.
- Zone-only parameters (RW_Z1 etc.) apply only inside their zone; outside, the
  global chain runs, with a warning.
- `apply_hc_correction=True` feeds NPHI_HC/RHOB_HC into the porosity step.
- A measured TEMP log (>= 50% finite coverage) is the temperature source; the
  modelled gradient fills its gaps.
- PERM_CC from the core poro-perm transform is written only where PHIE_CC lies
  within the porosity range of the plugs used in the fit; elsewhere NaN, with
  the count in the trace.
- Input PHIE_CC/PERM_CC are kept as PHIE_CC_INPUT/PERM_CC_INPUT instead of
  being removed.
- `DynamicWorkflowConfig.__post_init__`: the deprecated `lithology` alias only
  fills `lithology_domain` when it is None and is not mirrored back.
- StandardWorkflow: `matrix_dt`, `fluid_dt`, `matrix_density` default to None
  (from matrix_type); explicit values win; `rho_cl` is used.

### Electrical, permeability, saturation, clay volume

- Rxo/Rt Rw method reads RXO first, then RSHA, RESS.
- `perm_fallback_method` is honoured: perm_method, fallback, then the
  automatic order.
- Reversed SP polarity (SP-GR correlation > 0.3 when GR exists) swaps the
  auto-picked baselines, with a warning.
- Dual Water uses the Clavier-Coates-Dumanoir form (Swb = Vcl*phi_cl/phi_t).
- No VCL curve: PorosityModule writes PHIE = PHIT with a warning;
  SaturationModule routes to Archie with a warning.
- `north_sea_shaly_sand` and `nigeria_schlumberger` return NaN where the shale
  conductivity term alone exceeds 1/Rt.
- RmfModule reads RMFT/MFST for the Rmf measurement temperature.
- `rw_from_ssp` no longer clips; the workflow rejects the SP route when half or
  more of the values lie outside [0.01, 10] ohm.m.

### Evaluation

- S_V integrates from surface (`rho_overburden=2.3` g/cc above the log top,
  `water_depth_m=0`); PP_HYDROSTATIC uses the same water depth and no implied
  seawater column; SH_MIN uses that absolute S_V.
- Internal RHOB gaps are interpolated for the S_V integral.
- Trend-zone gradient threshold is per metre.
- Zone/formation HCPV and BVW use the summed, thickness-weighted forms.
- HCPV, zone summary, pay averages and BvwModule prefer PHIE_CC/PERM_CC;
  `BvwModule.add_to_well` does not overwrite an existing BVW.
- NetPayModule uses `cutoff_family`.
- `calculate_hcpv_from_well` counts net_pay_ft over pay samples with finite
  PHIE and SW.

### I/O, units, lithology

- LAS export outside benchmark reproduction writes `%.6g`.
- RDEP alias ranking: AT90 first. HCGR -> CGR.
- Core SWC/SOC in percent are converted to v/v by `standardize_curves`.
- `resolve_from_core_matrix`: median core grain density when present,
  otherwise the vote needs confidence >= 0.8.
- PHIE_CC is removed from the core-porosity sources.
- Multi-mineral solver and N-D endpoints use the log density response for
  evaporites.
- `harmonic_mean`/`geometric_mean`: zeros give 0; negatives excluded with a
  warning.
- Zone/ZoneSet: top-inclusive, bottom-exclusive boundaries.
- Elastic properties: impossible values are NaN; XRF siderite sulphur
  threshold follows the units; `bin_cumulative` skips NaN bins.

### Quality control

- DCAL gets a differential-caliper rule, -5 to +5 in.
- Value-range QC converts recognised units (NPHI pu, CALI mm, DT us/m) before
  screening; +/-inf under clip rules become NaN.
- WellboreQC roughness threshold is per metre; the washout threshold is used
  (CALI - BS > threshold).
- `create_saturation_height_plot` labels the points 'data' unless
  `data_label` is given.

## Deferred (no effect on results)

- Validation ranges for implausible settings; casing flag; `settings_used`
  cutoffs; evaporite label wording; SP Rw per-sample basis; sonic frame
  uncertainty; `include_global` default; peer-well calibration gating;
  GAS_CORRECTION_EFFECT text; VclModule P5/P95 endpoints (the intentional
  [20,100] window stays).
- Teapot aliases; tops without a unit; depth warnings for non-LAS inputs;
  ':' in LAS parameters; CSV leading zeros; `identify_lithology_zone`; FMI
  generic grouping.
- MCP session isolation, resource limits, not-assessable bad hole, NTG
  naming, ml pins, API doc coverage.
- `calculate_porosity` rhob_fluid bounds; FluidProperties HI table; curve
  arithmetic units; tops suffix warning; CSV codebook; COSTA package notes;
  reserved file names; crossplot limits, matrix point, fit-line style,
  header layout; custom-step parameter form; quick-look thresholds; header
  LITH mapping; audit reason text.

## Left to the maintainer (not code)

- Permission documents for GeoLink, Norsk Hydro, RealPore and EpsLog data;
  the License-Expression versus the bundled data licences; Teapot and
  Taranaki terms before bundling.
- Public repository/install route (README and docs URLs), changelog date.
